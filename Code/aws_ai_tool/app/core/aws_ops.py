import boto3
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Consider loading region, AMI ID, key pair name, security group from config or environment variables
AWS_REGION = "us-east-1" # Example region, make configurable
DEFAULT_AMI_ID = "ami-xxxxxxxxxxxxxxxxx" # Example Ubuntu AMI, make configurable
DEFAULT_KEY_NAME = "your-key-pair-name" # Make configurable
DEFAULT_SECURITY_GROUP_IDS = ["sg-xxxxxxxxxxxxxxxxx"] # Make configurable

def get_ec2_client():
    """Creates and returns an EC2 client.
    Handles credential sourcing via standard Boto3 mechanisms
    (environment variables, shared credential file, AWS config file, IAM role).
    """
    try:
        # Explicitly specify region, though Boto3 can often infer it
        client = boto3.client('ec2', region_name=AWS_REGION)
        # Perform a simple check to ensure credentials are valid
        client.describe_regions()
        logger.info(f"Successfully created EC2 client for region {AWS_REGION}")
        return client
    except ClientError as e:
        logger.error(f"Failed to create EC2 client: {e}. Check AWS credentials and configuration.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred creating EC2 client: {e}")
        raise

def get_ec2_resource():
    """Creates and returns an EC2 resource."""
    try:
        resource = boto3.resource('ec2', region_name=AWS_REGION)
        logger.info(f"Successfully created EC2 resource for region {AWS_REGION}")
        return resource
    except ClientError as e:
        logger.error(f"Failed to create EC2 resource: {e}. Check AWS credentials and configuration.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred creating EC2 resource: {e}")
        raise


def launch_ec2_instance(instance_type='t2.micro', ami_id=DEFAULT_AMI_ID, key_name=DEFAULT_KEY_NAME, security_group_ids=DEFAULT_SECURITY_GROUP_IDS, user_data=None, tags=None):
    """
    Launches a new EC2 instance.

    Args:
        instance_type (str): The EC2 instance type (e.g., 't2.micro', 'g4dn.xlarge').
        ami_id (str): The ID of the Amazon Machine Image to use.
        key_name (str): The name of the EC2 key pair for SSH access.
        security_group_ids (list): A list of security group IDs.
        user_data (str, optional): Script to run on instance launch. Defaults to None.
        tags (list, optional): A list of tags (dict format: [{'Key': 'Name', 'Value': 'MyInstance'}]) to apply. Defaults to None.

    Returns:
        str: The ID of the launched instance, or None if launch failed.
    """
    ec2 = get_ec2_resource()
    if not ec2:
        return None

    launch_args = {
        'ImageId': ami_id,
        'InstanceType': instance_type,
        'KeyName': key_name,
        'SecurityGroupIds': security_group_ids,
        'MinCount': 1,
        'MaxCount': 1,
    }

    if user_data:
        launch_args['UserData'] = user_data

    if tags:
        launch_args['TagSpecifications'] = [{'ResourceType': 'instance', 'Tags': tags}]

    try:
        logger.info(f"Attempting to launch EC2 instance with type {instance_type} using AMI {ami_id}...")
        instances = ec2.create_instances(**launch_args)
        instance = instances[0]
        logger.info(f"Successfully initiated launch for instance {instance.id}. Waiting for it to run...")

        # Wait for the instance to enter the 'running' state
        instance.wait_until_running()
        instance.reload() # Reload attributes after waiting

        logger.info(f"Instance {instance.id} is now running at IP: {instance.public_ip_address}")
        return instance.id
    except ClientError as e:
        logger.error(f"Failed to launch EC2 instance: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during instance launch: {e}")
        return None

def get_instance_status(instance_id):
    """
    Gets the status and details of a specific EC2 instance.

    Args:
        instance_id (str): The ID of the EC2 instance.

    Returns:
        dict: A dictionary containing instance details (status, IP address, etc.), or None if not found or error.
    """
    ec2 = get_ec2_client()
    if not ec2:
        return None

    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
        if not response['Reservations']:
            logger.warning(f"Instance {instance_id} not found.")
            return None

        instance_info = response['Reservations'][0]['Instances'][0]
        status = {
            'id': instance_id,
            'state': instance_info['State']['Name'],
            'public_ip': instance_info.get('PublicIpAddress'),
            'private_ip': instance_info.get('PrivateIpAddress'),
            'launch_time': instance_info.get('LaunchTime')
        }
        logger.info(f"Status for instance {instance_id}: {status['state']}")
        return status
    except ClientError as e:
        # Handle specific error if instance is not found after launch attempt
        if 'InvalidInstanceID.NotFound' in str(e):
             logger.warning(f"Instance {instance_id} not found (describe_instances).")
             return None
        logger.error(f"Failed to get status for instance {instance_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred getting instance status: {e}")
        return None


def terminate_ec2_instance(instance_id):
    """
    Terminates a specific EC2 instance.

    Args:
        instance_id (str): The ID of the EC2 instance to terminate.

    Returns:
        bool: True if termination was successfully initiated, False otherwise.
    """
    ec2 = get_ec2_client()
    if not ec2:
        return False

    try:
        logger.warning(f"Attempting to terminate instance {instance_id}...")
        response = ec2.terminate_instances(InstanceIds=[instance_id])
        state_change = response['TerminatingInstances'][0]
        logger.info(f"Termination initiated for instance {instance_id}. Current state: {state_change['CurrentState']['Name']}, Previous state: {state_change['PreviousState']['Name']}")
        return True
    except ClientError as e:
        logger.error(f"Failed to terminate instance {instance_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during instance termination: {e}")
        return False

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Note: Running this directly requires AWS credentials to be configured
    # where Boto3 can find them (e.g., environment variables, ~/.aws/credentials)
    print("Testing AWS Ops Module...")
    # test_instance_id = launch_ec2_instance(instance_type='t2.micro') # Be careful launching instances!
    test_instance_id = "i-xxxxxxxxxxxxxxxxx" # Replace with a real ID for status/terminate tests

    if test_instance_id:
        print(f"Launched/Using Instance ID: {test_instance_id}")
        import time
        time.sleep(10) # Give time for status to potentially update
        status = get_instance_status(test_instance_id)
        print(f"Instance Status: {status}")

        # time.sleep(5)
        # print("Attempting termination...")
        # terminated = terminate_ec2_instance(test_instance_id)
        # print(f"Termination successful: {terminated}")
    else:
        print("Could not launch or use test instance.")
