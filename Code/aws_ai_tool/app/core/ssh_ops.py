import paramiko
import logging
import os
import time
from scp import SCPClient, SCPException # Using scp library for easier file transfers over paramiko

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_SSH_USER = "ubuntu" # Common user for Ubuntu AMIs
# Key file path should be configurable, perhaps fetched from app config
DEFAULT_KEY_FILENAME = os.path.expanduser("~/.ssh/your-key-pair-name.pem") # Make configurable

def get_ssh_key(key_filename=DEFAULT_KEY_FILENAME):
    """Loads the private SSH key, trying different key types if necessary."""
    if not os.path.exists(key_filename):
        logger.error(f"SSH key file not found at {key_filename}. Please check the path.")
        return None
    
    # Check file permissions (should be 0600 or 0400 on Unix systems)
    if os.name == 'posix':  # Unix-like systems
        file_mode = os.stat(key_filename).st_mode & 0o777
        if file_mode != 0o600 and file_mode != 0o400:
            logger.warning(f"SSH key file {key_filename} has permissions {oct(file_mode)} which may be too permissive. "
                          f"Consider changing to 0600 with: chmod 600 {key_filename}")
    
    # Try different key types in sequence
    key_types = [
        ('RSA', lambda f: paramiko.RSAKey.from_private_key_file(f)),
        ('DSS', lambda f: paramiko.DSSKey.from_private_key_file(f)),
        ('ECDSA', lambda f: paramiko.ECDSAKey.from_private_key_file(f)),
        ('Ed25519', lambda f: paramiko.Ed25519Key.from_private_key_file(f)),
    ]
    
    for key_type, key_loader in key_types:
        try:
            key = key_loader(key_filename)
            logger.info(f"Successfully loaded {key_type} SSH private key from {key_filename}")
            return key
        except paramiko.PasswordRequiredException:
            logger.error(f"SSH key file {key_filename} is encrypted and requires a password (not supported).")
            return None
        except paramiko.SSHException:
            # This key type didn't work, try the next one
            continue
        except Exception as e:
            # Log the error but continue trying other key types
            logger.warning(f"Failed to load {key_type} SSH key from {key_filename}: {e}")
    
    # If we get here, all key types failed
    logger.error(f"Failed to load SSH key {key_filename} as any supported type (RSA, DSS, ECDSA, Ed25519)")
    
    # Try to read the file to provide more diagnostic information
    try:
        with open(key_filename, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('-----BEGIN '):
                logger.error(f"Key file does not appear to be in PEM format. First line: {first_line[:30]}...")
            else:
                logger.error(f"Key appears to be in PEM format but could not be loaded. Header: {first_line}")
    except Exception as e:
        logger.error(f"Could not read key file for diagnostics: {e}")
    
    return None

def create_ssh_client(hostname, username=DEFAULT_SSH_USER, key_filename=DEFAULT_KEY_FILENAME, retries=3, delay=5):
    """
    Creates and connects an SSH client to the specified host.

    Args:
        hostname (str): The public IP address or DNS name of the EC2 instance.
        username (str): The username to connect as (e.g., 'ubuntu', 'ec2-user').
        key_filename (str): Path to the private SSH key file.
        retries (int): Number of times to retry connection on failure.
        delay (int): Seconds to wait between retries.

    Returns:
        paramiko.SSHClient: An active SSH client object, or None if connection fails.
    """
    if not hostname:
        logger.error("Cannot create SSH client: hostname is missing.")
        return None

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Automatically add host key (less secure, ok for this context)
    # Consider using WarningPolicy or RejectPolicy in production with known_hosts management

    private_key = get_ssh_key(key_filename)
    if not private_key:
        return None

    for attempt in range(retries):
        try:
            logger.info(f"Attempting SSH connection to {username}@{hostname} (Attempt {attempt + 1}/{retries})...")
            client.connect(hostname=hostname, username=username, pkey=private_key, timeout=10)
            logger.info(f"SSH connection established successfully to {hostname}.")
            return client
        except paramiko.AuthenticationException:
            logger.error(f"Authentication failed for {username}@{hostname}. Check username and key.")
            return None # No point retrying on auth failure
        except (paramiko.SSHException, TimeoutError, OSError) as e:
            logger.warning(f"Could not connect to {hostname} (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to connect to {hostname} after {retries} attempts.")
                return None
        except Exception as e:
             logger.error(f"An unexpected error occurred during SSH connection to {hostname}: {e}")
             return None # No point retrying on unexpected errors

    return None # Should not be reached, but added for clarity

def execute_remote_command(client, command, timeout=None):
    """
    Executes a command on the remote host via the established SSH client.

    Args:
        client (paramiko.SSHClient): An active SSH client object.
        command (str): The command string to execute.
        timeout (int, optional): Timeout in seconds for the command execution. Defaults to None (no timeout).

    Returns:
        tuple: (stdout, stderr, exit_code)
               stdout (str): Standard output from the command.
               stderr (str): Standard error from the command.
               exit_code (int): Exit code of the command (-1 if error before execution).
    """
    if not client:
        logger.error("Cannot execute command: SSH client is not valid.")
        return "", "SSH client not available", -1

    try:
        logger.info(f"Executing remote command: {command}")
        stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_code = stdout.channel.recv_exit_status() # Wait for command to finish and get exit code
        stdout_data = stdout.read().decode('utf-8').strip()
        stderr_data = stderr.read().decode('utf-8').strip()

        if exit_code == 0:
            logger.info(f"Command executed successfully. Exit code: {exit_code}")
            # logger.debug(f"Stdout:\n{stdout_data}") # Optionally log stdout
        else:
            logger.warning(f"Command finished with non-zero exit code: {exit_code}")
            logger.warning(f"Stderr:\n{stderr_data}")
            # logger.debug(f"Stdout:\n{stdout_data}") # Optionally log stdout even on error

        return stdout_data, stderr_data, exit_code
    except paramiko.SSHException as e:
        logger.error(f"SSH error during command execution ('{command}'): {e}")
        return "", f"SSH error: {e}", -1
    except Exception as e:
        logger.error(f"An unexpected error occurred executing command ('{command}'): {e}")
        return "", f"Unexpected error: {e}", -1


def upload_file(client, local_path, remote_path, recursive=False):
    """
    Uploads a local file or directory to the remote host using SCP.

    Args:
        client (paramiko.SSHClient): An active SSH client object.
        local_path (str): The path to the local file or directory.
        remote_path (str): The destination path on the remote host.
        recursive (bool): Whether to upload directories recursively. Defaults to False.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    if not client:
        logger.error("Cannot upload file: SSH client is not valid.")
        return False
    if not os.path.exists(local_path):
        logger.error(f"Cannot upload file: Local path '{local_path}' does not exist.")
        return False

    try:
        # Setup SCP client
        with SCPClient(client.get_transport()) as scp:
            logger.info(f"Uploading '{local_path}' to '{remote_path}' (recursive={recursive})...")
            scp.put(local_path, remote_path, recursive=recursive)
            logger.info("Upload successful.")
            return True
    except SCPException as e:
        logger.error(f"SCP error during upload of '{local_path}' to '{remote_path}': {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during upload: {e}")
        return False

def download_file(client, remote_path, local_path, recursive=False):
    """
    Downloads a remote file or directory to the local machine using SCP.

    Args:
        client (paramiko.SSHClient): An active SSH client object.
        remote_path (str): The path to the file or directory on the remote host.
        local_path (str): The destination path on the local machine.
        recursive (bool): Whether to download directories recursively. Defaults to False.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    if not client:
        logger.error("Cannot download file: SSH client is not valid.")
        return False

    # Ensure local directory exists
    local_dir = os.path.dirname(local_path)
    if local_dir and not os.path.exists(local_dir):
        try:
            os.makedirs(local_dir)
            logger.info(f"Created local directory: {local_dir}")
        except OSError as e:
            logger.error(f"Failed to create local directory '{local_dir}': {e}")
            return False

    try:
        # Setup SCP client
        with SCPClient(client.get_transport()) as scp:
            logger.info(f"Downloading '{remote_path}' to '{local_path}' (recursive={recursive})...")
            scp.get(remote_path, local_path, recursive=recursive)
            logger.info("Download successful.")
            return True
    except SCPException as e:
        # Check if the error is because the remote file doesn't exist
        # Note: SCPException messages might vary, this is an example check
        if "No such file or directory" in str(e):
             logger.warning(f"Remote path '{remote_path}' not found for download.")
        else:
            logger.error(f"SCP error during download of '{remote_path}' to '{local_path}': {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during download: {e}")
        return False

# Example Usage (for testing purposes)
if __name__ == '__main__':
    print("Testing SSH Ops Module...")
    # --- Configuration for Testing ---
    # Replace with a real IP/DNS and ensure the key path is correct
    # and the corresponding EC2 instance is running and accessible.
    TEST_HOSTNAME = "YOUR_EC2_INSTANCE_IP_OR_DNS"
    TEST_KEY_PATH = DEFAULT_KEY_FILENAME # Use default or specify another path

    if TEST_HOSTNAME == "YOUR_EC2_INSTANCE_IP_OR_DNS":
        print("Please update TEST_HOSTNAME in the script before running tests.")
    else:
        ssh_client = create_ssh_client(TEST_HOSTNAME, key_filename=TEST_KEY_PATH)

        if ssh_client:
            print("\n--- Testing Command Execution ---")
            stdout, stderr, exit_code = execute_remote_command(ssh_client, "echo 'Hello from remote!'")
            print(f"Exit Code: {exit_code}")
            print(f"Stdout: {stdout}")
            print(f"Stderr: {stderr}")

            stdout, stderr, exit_code = execute_remote_command(ssh_client, "ls -l /home/ubuntu")
            print(f"\nls -l /home/ubuntu:\nExit Code: {exit_code}\nStdout:\n{stdout}\nStderr:\n{stderr}")

            stdout, stderr, exit_code = execute_remote_command(ssh_client, "this_command_does_not_exist")
            print(f"\nNon-existent command:\nExit Code: {exit_code}\nStdout:\n{stdout}\nStderr:\n{stderr}")


            print("\n--- Testing File Upload ---")
            # Create a dummy local file
            local_test_file = "local_test_upload.txt"
            remote_test_file = f"/tmp/{local_test_file}" # Use /tmp on remote
            with open(local_test_file, "w") as f:
                f.write("This is a test file for SCP upload.\n")
            print(f"Created local file: {local_test_file}")

            if upload_file(ssh_client, local_test_file, remote_test_file, recursive=False):
                print("Upload successful.")
                # Verify upload with ls
                stdout, stderr, exit_code = execute_remote_command(ssh_client, f"ls -l {remote_test_file}")
                print(f"Remote file details:\n{stdout}")
            else:
                print("Upload failed.")

            print("\n--- Testing File Download (Single File) ---")
            local_download_path = "downloaded_remote_file.txt"
            if download_file(ssh_client, remote_test_file, local_download_path, recursive=False):
                print(f"Download successful to {local_download_path}")
                # Verify content
                if os.path.exists(local_download_path):
                    with open(local_download_path, "r") as f:
                        print(f"Content of downloaded file:\n{f.read()}")
                    os.remove(local_download_path) # Clean up downloaded file
                else:
                    print("Downloaded file not found locally.")
            else:
                print("Single file download failed.")

            # Test directory download
            print("\n--- Testing Directory Download ---")
            remote_test_dir = "/tmp/test_remote_dir_download"
            local_test_dir_download = "downloaded_remote_dir"
            # Create some files in the remote dir
            execute_remote_command(ssh_client, f"mkdir -p {remote_test_dir}")
            execute_remote_command(ssh_client, f"echo 'file1' > {remote_test_dir}/file1.txt")
            execute_remote_command(ssh_client, f"echo 'file2' > {remote_test_dir}/file2.txt")
            print(f"Created remote directory {remote_test_dir} with files.")

            if download_file(ssh_client, remote_test_dir, local_test_dir_download, recursive=True):
                print(f"Directory download successful to {local_test_dir_download}")
                # Verify local content
                if os.path.exists(local_test_dir_download) and os.path.isdir(local_test_dir_download):
                    print(f"Contents of {local_test_dir_download}: {os.listdir(local_test_dir_download)}")
                    # Clean up local dir
                    import shutil
                    shutil.rmtree(local_test_dir_download)
                else:
                    print(f"Local directory {local_test_dir_download} not found.")
            else:
                 print("Directory download failed.")

            # Clean up remote file and dir
            print("\n--- Cleaning up remote test file and directory ---")
            stdout, stderr, exit_code = execute_remote_command(ssh_client, f"rm -rf {remote_test_file} {remote_test_dir}")
            print(f"Remote cleanup exit code: {exit_code}")

            # Clean up local file
            os.remove(local_test_file)
            print(f"Removed local file: {local_test_file}")


            # Close the client connection
            ssh_client.close()
            print("\nSSH connection closed.")
        else:
            print("Failed to establish SSH connection.")
