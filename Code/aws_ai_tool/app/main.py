from flask import Blueprint, render_template, request, url_for, flash, redirect, current_app, Response, jsonify
import os
import datetime
import json # For handling hyperparameters
import time # For timestamps
from werkzeug.utils import secure_filename # For secure file uploads
from .core import db as database # Use alias to avoid confusion with db variable
from .core import ssh_ops # Import SSH operations module

# Define the blueprint: 'main' is the name of this blueprint.
# We are telling Flask that this blueprint exists.
main = Blueprint('main', __name__)

# Route for the main page (Job Management)
@main.route('/')
def index():
    """Renders the main Job Management page, fetching job data and available models."""
    db = database.get_db()
    active_jobs = []
    completed_experiments = []
    available_models = []
    try:
        # Fetch available model scripts from upload folder
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_folder):
            try:
                available_models = [f for f in os.listdir(upload_folder) if f.endswith('.py') and os.path.isfile(os.path.join(upload_folder, f))]
                available_models.sort()
            except OSError as e:
                flash(f"Error accessing upload folder '{upload_folder}': {e}", "danger")
        else:
             # Don't flash warning on index page load if folder just doesn't exist yet
             pass

        # Fetch jobs that are potentially still running
        active_jobs_cursor = db.execute(
            "SELECT * FROM experiments WHERE status NOT IN ('Completed', 'Failed') ORDER BY created_at DESC"
        )
        active_jobs = active_jobs_cursor.fetchall()

        # Fetch jobs that have finished
        completed_experiments_cursor = db.execute(
            "SELECT * FROM experiments WHERE status IN ('Completed', 'Failed') ORDER BY end_time DESC, created_at DESC"
        )
        completed_experiments = completed_experiments_cursor.fetchall()

    except database.get_db().Error as e:
        flash(f"Database error fetching experiments: {e}", "danger")

    return render_template('index.html',
                           active_jobs=active_jobs,
                           completed_experiments=completed_experiments,
                           available_models=available_models) # Pass models to template

# Routes for other tabs
@main.route('/models')
def model_management():
    """Renders the Model Management page, listing uploaded scripts."""
    upload_folder = current_app.config['UPLOAD_FOLDER']
    models = []
    if os.path.exists(upload_folder):
        try:
            # List .py files in the upload folder
            models = [f for f in os.listdir(upload_folder) if f.endswith('.py') and os.path.isfile(os.path.join(upload_folder, f))]
            # Sort for consistent display
            models.sort()
        except OSError as e:
            flash(f"Error accessing upload folder '{upload_folder}': {e}", "danger")
    else:
        flash(f"Upload folder '{upload_folder}' does not exist.", "warning")

    return render_template('models.html', models=models)

@main.route('/inference')
def inference():
    """Renders the Inference & Comparison page, listing completed experiments."""
    db = database.get_db()
    trained_models = [] # This will hold experiments considered 'trained'
    try:
        # Fetch experiments that completed successfully
        # We might refine this later based on whether results were downloaded
        comp_exp_cursor = db.execute(
            "SELECT experiment_id, name, model_script FROM experiments WHERE status = 'Completed' ORDER BY end_time DESC, created_at DESC"
        )
        trained_models = comp_exp_cursor.fetchall()
    except database.get_db().Error as e:
        flash(f"Database error fetching trained models: {e}", "danger")

    return render_template('inference.html', trained_models=trained_models)

@main.route('/settings', methods=['GET'])
def settings():
    """Renders the Settings page, loading current config values."""
    # Load relevant settings from Flask config
    # Use .get() to avoid errors if a key is somehow missing
    current_settings = {
        'SSH_HOSTNAME': current_app.config.get('SSH_HOSTNAME', ''),
        'SSH_USERNAME': current_app.config.get('SSH_USERNAME', 'ubuntu'),
        # Expand ~ in the key path for display
        'SSH_KEY_PATH': os.path.expanduser(current_app.config.get('SSH_KEY_PATH', '~/.ssh/id_rsa')),
        'upload_folder': current_app.config.get('UPLOAD_FOLDER', 'uploads'),
        'results_folder': current_app.config.get('RESULTS_FOLDER', 'results'),
        'database_path': current_app.config.get('DATABASE', 'experiments.db'),
        # Add AWS settings back if needed later
        # 'aws_region': current_app.config.get('AWS_REGION', 'us-east-1'),
        # 'aws_key_name': current_app.config.get('AWS_KEY_NAME', ''),
        # 'aws_security_groups': ','.join(current_app.config.get('AWS_SECURITY_GROUP_IDS', [])),
        # 'default_ami_id': current_app.config.get('DEFAULT_AMI_ID', ''),
    }
    return render_template('settings.html', settings=current_settings)

# Routes for form submissions
@main.route('/launch_job', methods=['POST'])
def launch_job():
    """Handles the form submission for launching a new job."""
    exp_name = request.form.get('experiment_name')
    model_script_name = request.form.get('model_script')
    hyperparams_str = request.form.get('hyperparameters', '{}') # Default to empty JSON string
    data_source = request.form.get('data_source', '') # Path or description
    # instance_type = request.form.get('instance_type') # Not used when connecting to existing instance

    # --- Validation ---
    if not exp_name or not model_script_name:
        flash("Experiment Name and Model Script are required.", "danger")
        return redirect(url_for('main.index'))

    # Validate hyperparameters format (simple check for now)
    try:
        # Try parsing as JSON, but store as string in DB for flexibility
        json.loads(hyperparams_str)
    except json.JSONDecodeError:
        flash("Hyperparameters must be valid JSON.", "danger")
        return redirect(url_for('main.index'))

    # Check if selected model script exists
    model_script_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(model_script_name))
    if not os.path.exists(model_script_path):
         flash(f"Selected model script '{model_script_name}' not found in uploads.", "danger")
         return redirect(url_for('main.index'))

    # --- Get SSH Config ---
    hostname = current_app.config.get('SSH_HOSTNAME')
    username = current_app.config.get('SSH_USERNAME')
    key_path = os.path.expanduser(current_app.config.get('SSH_KEY_PATH')) # Expand ~

    if not hostname or not username or not key_path:
        flash("SSH connection details (Hostname, Username, Key Path) are not configured in Settings.", "danger")
        return redirect(url_for('main.settings'))
    if not os.path.exists(key_path):
         flash(f"SSH key file not found at '{key_path}'. Please check Settings.", "danger")
         return redirect(url_for('main.settings'))

    # --- Record initial job in DB ---
    db = database.get_db()
    cursor = None
    experiment_id = None
    error_details = None  # Variable to track specific error details
    
    try:
        cursor = db.execute(
            "INSERT INTO experiments (name, model_script, data_source, hyperparameters, status, start_time) VALUES (?, ?, ?, ?, ?, ?)",
            (exp_name, model_script_name, data_source, hyperparams_str, 'Setting Up', datetime.datetime.now())
        )
        db.commit()
        experiment_id = cursor.lastrowid
        current_app.logger.info(f"Experiment '{exp_name}' created (ID: {experiment_id}). Attempting SSH connection...")
        flash(f"Experiment '{exp_name}' created (ID: {experiment_id}). Attempting SSH connection...", "info")
    except database.get_db().Error as e:
        error_msg = f"Database error creating experiment: {e}"
        current_app.logger.error(error_msg)
        flash(error_msg, "danger")
        if cursor:
            db.rollback() # Rollback on error
        return redirect(url_for('main.index'))
    finally:
        # Note: Don't close DB connection here, Flask handles it per request context
        pass

    # --- SSH Connection and Setup ---
    ssh_client = None
    try:
        # Define remote paths (make configurable later if needed)
        remote_base_dir = f"/home/{username}/ai_tool_experiments"
        remote_exp_dir = f"{remote_base_dir}/{experiment_id}_{secure_filename(exp_name)}"
        remote_script_path = f"{remote_exp_dir}/{secure_filename(model_script_name)}"
        remote_output_dir = f"{remote_exp_dir}/output"
        remote_data_dir = f"{remote_exp_dir}/data" # Placeholder for data upload later
        
        # Establish SSH connection
        current_app.logger.info(f"Attempting to establish SSH connection to {hostname}...")
        ssh_client = ssh_ops.create_ssh_client(hostname, username, key_path)
        if not ssh_client:
            error_details = f"Failed to establish SSH connection to {hostname}."
            raise Exception(error_details)

        # Create remote directories
        current_app.logger.info(f"Creating remote directories at {remote_exp_dir}...")
        stdout, stderr, exit_code = ssh_ops.execute_remote_command(ssh_client, f"mkdir -p {remote_exp_dir} {remote_output_dir} {remote_data_dir}")
        if exit_code != 0:
            error_details = f"Failed to create remote directories: {stderr}"
            raise Exception(error_details)

        # Update DB status
        db.execute("UPDATE experiments SET status = ? WHERE experiment_id = ?", ('Transferring Script', experiment_id))
        db.commit()
        current_app.logger.info("SSH connected. Creating directories and uploading script...")
        flash("SSH connected. Creating directories and uploading script...", "info")

        # Upload the model script
        current_app.logger.info(f"Uploading model script {model_script_path} to {remote_script_path}...")
        if not ssh_ops.upload_file(ssh_client, model_script_path, remote_script_path):
            error_details = f"Failed to upload model script from {model_script_path} to {remote_script_path}."
            raise Exception(error_details)

        # TODO: Upload data if a mechanism is implemented

        # --- Construct and Execute Training Command ---
        # Standardize command-line arguments for user scripts
        cmd_args = f"--output_dir {remote_output_dir}" # Always pass output dir

        # Determine the base data directory on the remote machine
        # Assuming the dataset is consistently at '~/dataset/thermal' on the remote machine
        remote_data_base_dir = f"/home/{username}/dataset/thermal" # Use the path identified from user's ls output

        # Add the data directory argument, pointing to the assumed location
        cmd_args += f" --data_dir {remote_data_base_dir}"
        current_app.logger.info(f"Using remote data directory: {remote_data_base_dir}")

        # Parse hyperparameters from JSON and add them as arguments
        try:
            hyperparams = json.loads(hyperparams_str)
            # Map specific hyperparameter names expected by srcnn_train.py
            # Add other relevant hyperparameters here if needed
            if 'epochs' in hyperparams:
                cmd_args += f" --epochs {hyperparams['epochs']}"
            if 'lr' in hyperparams:
                cmd_args += f" --lr {hyperparams['lr']}"
            elif 'learning_rate' in hyperparams: # Allow alternative name
                 cmd_args += f" --lr {hyperparams['learning_rate']}"
            if 'batch_size' in hyperparams:
                cmd_args += f" --batch_size {hyperparams['batch_size']}"

            # Log the hyperparameters being used
            current_app.logger.info(f"Parsed hyperparameters: {hyperparams}")
            current_app.logger.info(f"Constructed command arguments: {cmd_args}")

        except json.JSONDecodeError:
            current_app.logger.warning(f"Could not parse hyperparameters JSON: {hyperparams_str}. Proceeding without hyperparameters.")
            # Decide if you want to proceed without hyperparameters or fail
            # flash("Invalid JSON format for hyperparameters.", "danger")
            # return redirect(url_for('main.index'))
            pass # Continue without hyperparameters for now

        # Command to run the script in a tmux session, redirecting output
        log_file = f"{remote_output_dir}/training.log"
        tmux_session_name = f"job_{experiment_id}"

        # The training script itself now handles creating _SUCCESS or _FAILED markers
        training_command = (
            f"cd {remote_exp_dir} && " # Change to experiment directory
            # Ensure python3 environment has necessary packages (torch, torchvision, etc.)
            f"python3 {secure_filename(model_script_name)} {cmd_args} > {log_file} 2>&1"
            # The script srcnn_train.py will create _SUCCESS or _FAILED in its output_dir
        )

        # Wrap the training command in a tmux session
        remote_command = f"tmux new-session -d -s {tmux_session_name} '{training_command}'"

        current_app.logger.info(f"Creating tmux session '{tmux_session_name}' for training job with command: {training_command}")

        # Update DB status
        db.execute("UPDATE experiments SET status = ? WHERE experiment_id = ?", ('Training', experiment_id))
        db.commit()
        current_app.logger.info("Script uploaded. Executing training command remotely...")
        flash("Script uploaded. Executing training command remotely...", "success")

        # Execute the command (non-blocking due to nohup/&)
        current_app.logger.info(f"Executing remote command: {remote_command}")
        stdout, stderr, exit_code = ssh_ops.execute_remote_command(ssh_client, remote_command)

        if exit_code != 0:
            # This exit code is for the nohup command itself, not the background script
            # Usually 0 unless nohup fails immediately
            error_details = f"Remote command execution might have had issues starting: {stderr}"
            current_app.logger.warning(error_details)
            flash(f"Warning: Remote command execution might have had issues starting. Check logs for details.", "warning")
        else:
            current_app.logger.info(f"Training job '{exp_name}' launched successfully in the background on {hostname}.")
            flash(f"Training job '{exp_name}' launched successfully in the background on {hostname}.", "success")

    except Exception as e:
        error_msg = f"Error during job launch: {e}"
        current_app.logger.error(error_msg)
        flash(error_msg, "danger")
        
        # Update DB status to Failed if an error occurred after creation
        if experiment_id:
            try:
                # Use the specific error_details if available, otherwise use the general exception
                error_message = error_details if error_details else str(e)
                db.execute(
                    "UPDATE experiments SET status = ?, end_time = ?, error_message = ? WHERE experiment_id = ?", 
                    ('Failed', datetime.datetime.now(), error_message, experiment_id)
                )
                db.commit()
                current_app.logger.info(f"Updated experiment {experiment_id} status to Failed with error: {error_message}")
            except database.get_db().Error as db_err:
                current_app.logger.error(f"Failed to update experiment status in DB: {db_err}")
                flash(f"Additionally, failed to update experiment status in DB: {db_err}", "danger")
    finally:
        if ssh_client:
            ssh_client.close()
            current_app.logger.info("SSH connection closed.")

    return redirect(url_for('main.index'))


@main.route('/upload_model', methods=['POST'])
def upload_model():
    """Handles model script uploads."""
    if 'model_file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(request.url) # Redirect back to the models page

    file = request.files['model_file']

    if file.filename == '':
        flash('No selected file.', 'warning')
        return redirect(url_for('main.model_management'))

    if file and file.filename.endswith('.py'):
        filename = secure_filename(file.filename) # Ensure filename is safe
        upload_folder = current_app.config['UPLOAD_FOLDER']

        # Ensure upload folder exists (should be created by __init__, but double-check)
        if not os.path.exists(upload_folder):
            try:
                os.makedirs(upload_folder)
            except OSError as e:
                 flash(f"Failed to create upload folder '{upload_folder}': {e}", "danger")
                 return redirect(url_for('main.model_management'))

        file_path = os.path.join(upload_folder, filename)

        try:
            file.save(file_path)
            flash(f'Model script "{filename}" uploaded successfully.', 'success')
        except Exception as e:
            flash(f'Error saving file "{filename}": {e}', 'danger')

        return redirect(url_for('main.model_management'))
    else:
        flash('Invalid file type. Please upload a Python (.py) script.', 'danger')
        return redirect(url_for('main.model_management'))


@main.route('/run_inference', methods=['POST'])
def run_inference():
    """Handles image upload and runs inference on the remote AWS instance."""
    if 'image_file' not in request.files:
        flash('No image file part in the request.', 'danger')
        return redirect(url_for('main.inference'))

    file = request.files['image_file']

    if file.filename == '':
        flash('No selected image file.', 'warning')
        return redirect(url_for('main.inference'))

    # Get selected trained model IDs
    trained_model_ids = request.form.getlist('trained_model_id')
    if not trained_model_ids:
        flash('No trained models selected.', 'warning')
        return redirect(url_for('main.inference'))

    # Basic check for image file types (can be expanded)
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tif', 'tiff'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        flash('Invalid image file type.', 'danger')
        return redirect(url_for('main.inference'))

    # Save uploaded image to a subfolder in static for display
    filename = secure_filename(file.filename)
    static_image_folder = os.path.join(current_app.static_folder, 'inference_images')
    if not os.path.exists(static_image_folder):
        os.makedirs(static_image_folder)

    image_path = os.path.join(static_image_folder, filename)
    original_image_url = None

    try:
        file.save(image_path)
        # Generate URL for the saved static file
        original_image_url = url_for('static', filename=f'inference_images/{filename}', _external=False)
        flash(f'Image "{filename}" uploaded successfully.', 'success')
    except Exception as e:
        flash(f'Error saving uploaded image: {e}', 'danger')
        return redirect(url_for('main.inference'))

    # Fetch trained models for the dropdown
    db = database.get_db()
    trained_models = []
    try:
        comp_exp_cursor = db.execute(
            "SELECT experiment_id, name, model_script FROM experiments WHERE status = 'Completed' ORDER BY end_time DESC, created_at DESC"
        )
        trained_models = comp_exp_cursor.fetchall()
    except database.get_db().Error as e:
        flash(f"Database error fetching trained models: {e}", "danger")
        return redirect(url_for('main.inference'))

    # Initialize results with the original image
    inference_results = [{'type': 'original', 'url': original_image_url, 'name': 'Original'}]
    
    # --- Get SSH Config ---
    hostname = current_app.config.get('SSH_HOSTNAME')
    username = current_app.config.get('SSH_USERNAME')
    key_path = os.path.expanduser(current_app.config.get('SSH_KEY_PATH'))

    if not hostname or not username or not key_path or not os.path.exists(key_path):
        flash("SSH connection details are missing or invalid in settings.", "danger")
        return render_template('inference.html',
                               trained_models=trained_models,
                               inference_results=inference_results)

    # Process each selected model
    for model_id in trained_model_ids:
        try:
            # Get experiment details
            exp_cursor = db.execute("SELECT * FROM experiments WHERE experiment_id = ?", (model_id,))
            exp = exp_cursor.fetchone()
            
            if not exp:
                flash(f"Experiment with ID {model_id} not found.", "danger")
                continue
                
            if exp['status'] != 'Completed':
                flash(f"Experiment '{exp['name']}' (ID: {model_id}) is not completed.", "warning")
                continue
                
            if not exp['results_path'] or not os.path.exists(exp['results_path']):
                flash(f"Results for experiment '{exp['name']}' (ID: {model_id}) not found locally.", "warning")
                continue
                
            # Look for model file in the results directory
            model_files = [f for f in os.listdir(exp['results_path']) 
                          if f.endswith('.pth') or f == 'dummy_model.pth']
            
            if not model_files:
                flash(f"No model file found for experiment '{exp['name']}' (ID: {model_id}).", "warning")
                continue
                
            model_file = model_files[0]  # Use the first model file found
            model_path = os.path.join(exp['results_path'], model_file)
            
            current_app.logger.info(f"Running inference with model {model_path} on image {image_path}")
            
            # Create SSH client
            ssh_client = ssh_ops.create_ssh_client(hostname, username, key_path)
            if not ssh_client:
                flash(f"Failed to establish SSH connection to {hostname}.", "danger")
                continue
                
            try:
                # Define remote paths
                timestamp = int(time.time())
                remote_base_dir = f"/home/{username}/ai_tool_inference"
                remote_model_dir = f"{remote_base_dir}/{model_id}"
                remote_input_path = f"{remote_base_dir}/input_{timestamp}_{filename}"
                remote_output_dir = f"{remote_base_dir}/output_{timestamp}"
                remote_output_path = f"{remote_output_dir}/{filename}"
                remote_script_path = f"{remote_base_dir}/remote_inference.py"
                
                # Create remote directories
                current_app.logger.info(f"Creating remote directories at {remote_base_dir}...")
                stdout, stderr, exit_code = ssh_ops.execute_remote_command(
                    ssh_client, f"mkdir -p {remote_base_dir} {remote_model_dir} {remote_output_dir}"
                )
                
                if exit_code != 0:
                    raise Exception(f"Failed to create remote directories: {stderr}")
                
                # Upload the input image
                current_app.logger.info(f"Uploading input image to {remote_input_path}...")
                if not ssh_ops.upload_file(ssh_client, image_path, remote_input_path):
                    raise Exception(f"Failed to upload input image to {remote_input_path}")
                
                # Upload the inference script if it doesn't exist
                local_script_path = os.path.join(current_app.config['UPLOAD_FOLDER'], '..', 'remote_inference.py')
                if os.path.exists(local_script_path):
                    current_app.logger.info(f"Uploading inference script to {remote_script_path}...")
                    if not ssh_ops.upload_file(ssh_client, local_script_path, remote_script_path):
                        raise Exception(f"Failed to upload inference script to {remote_script_path}")
                    
                    # Make the script executable
                    stdout, stderr, exit_code = ssh_ops.execute_remote_command(
                        ssh_client, f"chmod +x {remote_script_path}"
                    )
                    
                    if exit_code != 0:
                        raise Exception(f"Failed to make inference script executable: {stderr}")
                else:
                    current_app.logger.warning(f"Inference script not found at {local_script_path}")
                    raise Exception(f"Inference script not found at {local_script_path}")
                
                # Upload the model file if it doesn't exist on the remote
                remote_model_path = f"{remote_model_dir}/{model_file}"
                stdout, stderr, exit_code = ssh_ops.execute_remote_command(
                    ssh_client, f"test -f {remote_model_path}"
                )
                
                if exit_code != 0:
                    current_app.logger.info(f"Uploading model file to {remote_model_path}...")
                    if not ssh_ops.upload_file(ssh_client, model_path, remote_model_path):
                        raise Exception(f"Failed to upload model file to {remote_model_path}")
                else:
                    current_app.logger.info(f"Model file already exists at {remote_model_path}")
                
                # Run the inference script
                inference_command = (
                    f"python3 {remote_script_path} "
                    f"--model_path {remote_model_path} "
                    f"--input_path {remote_input_path} "
                    f"--output_path {remote_output_path}"
                )
                
                current_app.logger.info(f"Running inference command: {inference_command}")
                stdout, stderr, exit_code = ssh_ops.execute_remote_command(ssh_client, inference_command)
                
                if exit_code != 0:
                    raise Exception(f"Inference failed with exit code {exit_code}: {stderr}")
                
                current_app.logger.info(f"Inference stdout: {stdout}")
                
                # Download the output image
                static_output_folder = os.path.join(current_app.static_folder, 'inference_results')
                if not os.path.exists(static_output_folder):
                    os.makedirs(static_output_folder)
                
                output_filename = f"{exp['name']}_{model_id}_{filename}"
                local_output_path = os.path.join(static_output_folder, output_filename)
                
                current_app.logger.info(f"Downloading output image from {remote_output_path} to {local_output_path}...")
                if not ssh_ops.download_file(ssh_client, remote_output_path, local_output_path):
                    raise Exception(f"Failed to download output image from {remote_output_path}")
                
                # Generate URL for the output image
                output_image_url = url_for('static', filename=f'inference_results/{output_filename}', _external=False)
                
                # Add the result to the list
                inference_results.append({
                    'type': 'processed',
                    'url': output_image_url,
                    'name': f"Processed by {exp['name']} (ID: {model_id})"
                })
                
                flash(f"Inference with model '{exp['name']}' completed successfully.", "success")
                
                # Clean up remote files
                current_app.logger.info("Cleaning up remote files...")
                ssh_ops.execute_remote_command(
                    ssh_client, f"rm -f {remote_input_path} {remote_output_path}"
                )
                
            finally:
                # Close SSH connection
                ssh_client.close()
                current_app.logger.info("SSH connection closed.")
                
        except Exception as e:
            current_app.logger.error(f"Error during inference with model ID {model_id}: {e}")
            flash(f"Error during inference with model ID {model_id}: {e}", "danger")

    # Render the template with all results
    return render_template('inference.html',
                           trained_models=trained_models,
                           inference_results=inference_results)


@main.route('/save_settings', methods=['POST'])
def save_settings():
    """Handles saving application settings."""
    try:
        # Update Flask config in memory (won't persist across restarts yet)
        # TODO: Implement saving to a persistent config file (e.g., JSON, .env)
        # TODO: Add validation for paths and other inputs
        current_app.config['SSH_HOSTNAME'] = request.form.get('ssh_hostname', '')
        current_app.config['SSH_USERNAME'] = request.form.get('ssh_username', 'ubuntu')
        # Store the raw path, potentially with ~
        current_app.config['SSH_KEY_PATH'] = request.form.get('ssh_key_path', '~/.ssh/id_rsa')
        current_app.config['UPLOAD_FOLDER'] = request.form.get('upload_folder', 'uploads')
        current_app.config['RESULTS_FOLDER'] = request.form.get('results_folder', 'results')
        current_app.config['DATABASE'] = request.form.get('database_path', 'experiments.db')

        # Ensure new upload/results folders exist if changed
        if not os.path.exists(current_app.config['UPLOAD_FOLDER']):
            os.makedirs(current_app.config['UPLOAD_FOLDER'])
        if not os.path.exists(current_app.config['RESULTS_FOLDER']):
            os.makedirs(current_app.config['RESULTS_FOLDER'])

        flash("Settings updated in memory (will reset on restart). Persistent saving not yet implemented.", "success")
    except Exception as e:
        flash(f"Error saving settings: {e}", "danger")

    return redirect(url_for('main.settings'))

# Routes for actions mentioned in templates (e.g., view logs, terminate)
@main.route('/logs/<int:job_id>')
def view_logs(job_id):
    """Displays logs for a specific job by fetching from the remote server."""
    db = database.get_db()
    log_content = f"Logs for Job ID: {job_id}\n\n"
    exp = None
    try:
        exp_cursor = db.execute("SELECT * FROM experiments WHERE experiment_id = ?", (job_id,))
        exp = exp_cursor.fetchone()
    except database.get_db().Error as e:
        log_content += f"Database error fetching experiment details: {e}"
        return Response(log_content, mimetype='text/plain')

    if not exp:
        log_content += "Experiment not found in database."
        return Response(log_content, mimetype='text/plain')

    # Display stored error message if job failed
    if exp['status'] == 'Failed' and exp.get('error_message'):
        log_content += f"ERROR: {exp['error_message']}\n\n"
        log_content += "--- Full Log Content (if available) ---\n\n"

    # --- Get SSH Config ---
    hostname = current_app.config.get('SSH_HOSTNAME')
    username = current_app.config.get('SSH_USERNAME')
    key_path = os.path.expanduser(current_app.config.get('SSH_KEY_PATH'))

    if not hostname or not username or not key_path or not os.path.exists(key_path):
        log_content += "SSH connection details are missing or invalid in settings."
        return Response(log_content, mimetype='text/plain')

    # --- Construct remote paths ---
    remote_base_dir = f"/home/{username}/ai_tool_experiments"
    remote_exp_dir = f"{remote_base_dir}/{exp['experiment_id']}_{secure_filename(exp['name'])}"
    remote_output_dir = f"{remote_exp_dir}/output"
    remote_log_file = f"{remote_output_dir}/training.log"

    # --- Fetch log via SSH ---
    ssh_client = None
    try:
        current_app.logger.info(f"Attempting to fetch logs for job {job_id}...")
        ssh_client = ssh_ops.create_ssh_client(hostname, username, key_path, retries=1) # Fewer retries for log view
        if not ssh_client:
            raise Exception("Failed to establish SSH connection.")

        # Use 'tail' instead of 'cat' to avoid loading huge log files entirely into memory
        # Use a large number of lines to get most of the content for smaller logs
        stdout, stderr, exit_code = ssh_ops.execute_remote_command(ssh_client, f"tail -n 500 {remote_log_file}")

        if exit_code == 0:
            log_content += stdout if stdout else "(Log file is empty)"
        else:
            # Check common errors like file not found
            if "No such file or directory" in stderr:
                log_content += f"(Log file not found at {remote_log_file} - Job might be starting or failed early)\n\n"
                
                # Try to list the contents of the output directory to help diagnose path issues
                stdout_ls, stderr_ls, exit_code_ls = ssh_ops.execute_remote_command(
                    ssh_client, f"ls -la {remote_output_dir}"
                )
                
                if exit_code_ls == 0:
                    log_content += f"Contents of output directory ({remote_output_dir}):\n{stdout_ls}"
                else:
                    # If output directory doesn't exist, try to list the experiment directory
                    stdout_ls_exp, stderr_ls_exp, exit_code_ls_exp = ssh_ops.execute_remote_command(
                        ssh_client, f"ls -la {remote_exp_dir}"
                    )
                    
                    if exit_code_ls_exp == 0:
                        log_content += f"Output directory not found. Contents of experiment directory ({remote_exp_dir}):\n{stdout_ls_exp}"
                    else:
                        log_content += f"Neither log file nor directories could be found. The job may not have started properly."
            else:
                log_content += f"Error fetching log file via SSH (Exit Code: {exit_code}):\n{stderr}"

    except Exception as e:
        log_content += f"\nError during log fetching: {e}"
        current_app.logger.error(f"Error fetching logs for job {job_id}: {e}")
    finally:
        if ssh_client:
            ssh_client.close()
            current_app.logger.info(f"SSH connection closed for job {job_id} log view")

    # Return content as plain text
    return Response(log_content, mimetype='text/plain')

@main.route('/check_status/<int:job_id>')
def check_status(job_id):
    """Checks the status of a remote job and updates the database accordingly."""
    db = database.get_db()
    exp = None
    current_status = "Unknown"
    error_message = None
    
    try:
        exp_cursor = db.execute("SELECT * FROM experiments WHERE experiment_id = ?", (job_id,))
        exp = exp_cursor.fetchone()
    except database.get_db().Error as e:
        current_app.logger.error(f"Database error fetching experiment {job_id}: {e}")
        return jsonify({'status': 'Error', 'message': f"DB Error: {e}"}), 500

    if not exp:
        current_app.logger.warning(f"Experiment {job_id} not found in database")
        return jsonify({'status': 'Error', 'message': "Experiment not found"}), 404

    current_status = exp['status'] # Get status from DB first
    error_message = exp['error_message'] if 'error_message' in exp and exp['error_message'] is not None else None # Get any existing error message

    # Don't re-check if already completed/failed
    if current_status in ['Completed', 'Failed']:
        return jsonify({'status': current_status, 'error_message': error_message})

    # --- Get SSH Config ---
    hostname = current_app.config.get('SSH_HOSTNAME')
    username = current_app.config.get('SSH_USERNAME')
    key_path = os.path.expanduser(current_app.config.get('SSH_KEY_PATH'))

    if not hostname or not username or not key_path or not os.path.exists(key_path):
        # Cannot check status without SSH details
        current_app.logger.warning(f"SSH configuration missing for status check of job {job_id}")
        return jsonify({'status': current_status, 'message': 'SSH config missing'})

    # --- Construct remote paths ---
    remote_base_dir = f"/home/{username}/ai_tool_experiments"
    remote_exp_dir = f"{remote_base_dir}/{exp['experiment_id']}_{secure_filename(exp['name'])}"
    remote_output_dir = f"{remote_exp_dir}/output"
    remote_log_file = f"{remote_output_dir}/training.log"
    # Define marker files
    success_marker = f"{remote_output_dir}/_SUCCESS"
    failure_marker = f"{remote_output_dir}/_FAILED"

    # --- Check status via SSH ---
    ssh_client = None
    remote_status = current_status # Default to DB status if SSH fails
    download_failed = False
    
    try:
        current_app.logger.info(f"Checking status for job {job_id} via SSH...")
        ssh_client = ssh_ops.create_ssh_client(hostname, username, key_path, retries=1)
        if not ssh_client:
            raise Exception("SSH connection failed")

        # Check for completion markers first
        stdout_success, _, exit_code_success = ssh_ops.execute_remote_command(ssh_client, f"test -f {success_marker}")
        stdout_failure, _, exit_code_failure = ssh_ops.execute_remote_command(ssh_client, f"test -f {failure_marker}")

        if exit_code_success == 0:
            current_app.logger.info(f"Job {job_id} completed successfully (_SUCCESS marker found)")
            remote_status = 'Completed'
            # Clear any previous error message if job is now successful
            error_message = None
        elif exit_code_failure == 0:
            current_app.logger.info(f"Job {job_id} failed (_FAILED marker found)")
            remote_status = 'Failed'
            
            # If no error message is already set, try to get one from the log file
            if not error_message:
                current_app.logger.info(f"Attempting to retrieve error details from log for job {job_id}")
                # Try to read the _FAILED file first (it might contain the error message)
                stdout_failed_content, stderr_failed, exit_code_failed = ssh_ops.execute_remote_command(
                    ssh_client, f"cat {failure_marker}"
                )
                
                if exit_code_failed == 0 and stdout_failed_content.strip():
                    error_message = f"Training failed: {stdout_failed_content.strip()}"
                    current_app.logger.info(f"Retrieved error from _FAILED file: {error_message}")
                else:
                    # Try to get the last few lines of the log file
                    stdout_log_tail, stderr_log, exit_code_log = ssh_ops.execute_remote_command(
                        ssh_client, f"tail -n 10 {remote_log_file}"
                    )
                    
                    if exit_code_log == 0:
                        error_message = f"Training failed. Last log entries:\n{stdout_log_tail}"
                        current_app.logger.info(f"Retrieved error from log tail for job {job_id}")
                    else:
                        error_message = "Training failed but could not retrieve error details from log."
                        current_app.logger.warning(f"Could not retrieve log tail for failed job {job_id}: {stderr_log}")
        else:
            # If no markers, check if log file exists (implies Training or still running)
            stdout_log, _, exit_code_log = ssh_ops.execute_remote_command(ssh_client, f"test -f {remote_log_file}")
            if exit_code_log == 0 and current_status != 'Training':
                 # If log exists but DB status wasn't 'Training', update it
                 current_app.logger.info(f"Job {job_id} is now in Training status (log file exists)")
                 remote_status = 'Training'
            elif exit_code_log != 0 and current_status == 'Training':
                 # Log file disappeared or never created after being 'Training'? Assume failure.
                 current_app.logger.warning(f"Job {job_id} was in Training status but log file is missing - marking as Failed")
                 remote_status = 'Failed'
                 error_message = "Training failed: Log file disappeared during training."
            else:
                 # Otherwise, keep DB status (e.g., Setting Up, Transferring Script)
                 current_app.logger.info(f"Job {job_id} status unchanged: {current_status}")
                 remote_status = current_status

        # Fetch the latest log lines if the job is in Training status
        latest_log_lines = None
        if remote_status == 'Training':
            # Get the last 50 lines of the log file to show progress
            stdout_log_tail, stderr_log_tail, exit_code_log_tail = ssh_ops.execute_remote_command(
                ssh_client, f"tail -n 50 {remote_log_file}"
            )
            
            if exit_code_log_tail == 0 and stdout_log_tail:
                latest_log_lines = stdout_log_tail
                current_app.logger.debug(f"Retrieved latest log lines for job {job_id}")
        
        # --- Handle Completion ---
        if remote_status in ['Completed', 'Failed'] and current_status != remote_status:
            current_app.logger.info(f"Job {job_id} status changed to {remote_status}. Attempting to download results...")
            local_results_dir = os.path.join(current_app.config['RESULTS_FOLDER'], str(job_id))
            download_ok = ssh_ops.download_file(ssh_client, remote_output_dir, local_results_dir, recursive=True)

            if download_ok:
                current_app.logger.info(f"Results for job {job_id} downloaded to {local_results_dir}")
                # Update DB with final status, end time, results path, and error message if applicable
                if error_message:
                    db.execute(
                        "UPDATE experiments SET status = ?, end_time = ?, results_path = ?, error_message = ? WHERE experiment_id = ?",
                        (remote_status, datetime.datetime.now(), local_results_dir, error_message, job_id)
                    )
                else:
                    db.execute(
                        "UPDATE experiments SET status = ?, end_time = ?, results_path = ?, error_message = NULL WHERE experiment_id = ?",
                        (remote_status, datetime.datetime.now(), local_results_dir, job_id)
                    )
                db.commit()
            else:
                current_app.logger.error(f"Failed to download results for job {job_id}")
                download_failed = True
                # Append download failure to error message
                if error_message:
                    error_message = f"{error_message}\n[Result download failed]"
                else:
                    error_message = "[Result download failed]"
                
                # Update DB status with error message and note download failure
                db.execute(
                    "UPDATE experiments SET status = ?, end_time = ?, error_message = ? WHERE experiment_id = ?",
                    (remote_status, datetime.datetime.now(), error_message, job_id)
                )
                db.commit()

        # Update DB status if changed and not handled by completion logic above
        elif remote_status != current_status:
            current_app.logger.info(f"Updating job {job_id} status from {current_status} to {remote_status}")
            if error_message:
                db.execute(
                    "UPDATE experiments SET status = ?, error_message = ? WHERE experiment_id = ?", 
                    (remote_status, error_message, job_id)
                )
            else:
                db.execute(
                    "UPDATE experiments SET status = ? WHERE experiment_id = ?", 
                    (remote_status, job_id)
                )
            db.commit()

    except Exception as e:
        error_msg = f"Error checking status for job {job_id}: {e}"
        current_app.logger.error(error_msg)
        # Return current DB status if SSH check fails
        return jsonify({'status': current_status, 'message': error_msg})
    finally:
        if ssh_client:
            ssh_client.close()
            current_app.logger.info(f"SSH connection closed for job {job_id} status check")

    # Return the updated status, error message, and latest log lines if applicable
    response = {'status': remote_status}
    if error_message:
        response['error_message'] = error_message
    if download_failed:
        response['download_failed'] = True
    if latest_log_lines:
        response['latest_log_lines'] = latest_log_lines
        
    return jsonify(response)

# Temporary route for debugging - manually mark a job as completed
@main.route('/mark_completed/<int:job_id>')
def mark_completed(job_id):
    db = database.get_db()
    try:
        # Fetch the experiment to get its details
        exp_cursor = db.execute("SELECT * FROM experiments WHERE experiment_id = ?", (job_id,))
        exp = exp_cursor.fetchone()
        
        if not exp:
            flash(f"Experiment {job_id} not found.", "danger")
            return redirect(url_for('main.index'))
            
        # Define local results path (assuming it should exist)
        local_results_dir = os.path.join(current_app.config['RESULTS_FOLDER'], str(job_id))
        
        # Update the database
        db.execute(
            "UPDATE experiments SET status = ?, end_time = ?, results_path = ?, error_message = NULL WHERE experiment_id = ?",
            ('Completed', datetime.datetime.now(), local_results_dir, job_id)
        )
        db.commit()
        flash(f"Manually marked job {job_id} as 'Completed'.", "success")
        current_app.logger.info(f"Manually marked job {job_id} as 'Completed'.")
        
    except database.get_db().Error as e:
        flash(f"Database error marking job {job_id} as completed: {e}", "danger")
        current_app.logger.error(f"Database error marking job {job_id} as completed: {e}")
    except Exception as e:
        flash(f"Error marking job {job_id} as completed: {e}", "danger")
        current_app.logger.error(f"Error marking job {job_id} as completed: {e}")
        
    return redirect(url_for('main.index'))

# Add other routes for form submissions (launching jobs, uploading models, etc.) later
