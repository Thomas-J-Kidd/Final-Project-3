# AWS Training Manager for Deep Learning Models

This system allows you to deploy, run, and manage your deep learning training jobs on AWS. It includes tools for generating configuration variations, scheduling jobs, and retrieving results.

## Setup

1. Make sure you have the SSH key file at `~/.ssh/Thomas Kidd Deep Learning.pem`
2. Ensure the AWS server is running at the IP address in the scripts (3.140.193.132)
3. Ensure you have Python dependencies installed:
   ```bash
   pip install pyyaml
   ```

## Scripts Overview

### 1. aws_deploy.py

This script handles the low-level operations of deploying your code to AWS, running jobs, and retrieving results.

```bash
# Set up the AWS environment
python aws_deploy.py setup

# Run a training job
python aws_deploy.py run --config ThermalDenoising/conf_gopro_nafnet_rgb.yml --mode train

# Run a testing job
python aws_deploy.py run --config ThermalDenoising/conf_gopro_nafnet_rgb.yml --mode test

# Check status of running jobs
python aws_deploy.py status

# Retrieve results from AWS
python aws_deploy.py retrieve --config ThermalDenoising/conf_gopro_nafnet_rgb.yml
```

### 2. aws_train_manager.py

This higher-level script helps manage multiple configurations, create variations, and queue jobs.

```bash
# List all available configurations
python aws_train_manager.py list-configs

# Create variations of a base configuration
python aws_train_manager.py create-variations --base-config ThermalDenoising/conf_gopro_nafnet_rgb.yml --params "LR=0.0001,0.0002" "BATCH_SIZE=4,8"

# Queue a job for execution
python aws_train_manager.py queue --config ThermalDenoising/conf_gopro_nafnet_rgb.yml --mode train

# Process the next job in the queue
python aws_train_manager.py process

# Check status of all jobs
python aws_train_manager.py list-jobs
```

### 3. create_nafnet_experiment.py

This script creates a set of predefined variations of the NAFNET configuration for experimenting with different hyperparameters.

```bash
# Generate experiment configurations
python create_nafnet_experiment.py
```

## Workflow Example

Here's a complete workflow example to run experiments on AWS:

1. **Setup the AWS environment**:
   ```bash
   python aws_deploy.py setup
   ```

2. **Create experiment configurations**:
   ```bash
   # Either use predefined variations
   python create_nafnet_experiment.py
   
   # Or create custom variations
   python aws_train_manager.py create-variations --base-config ThermalDenoising/conf_gopro_nafnet_rgb.yml --params "LR=0.0001,0.0002,0.0003" "BATCH_SIZE=2,4,8"
   ```

3. **Queue jobs for training**:
   ```bash
   # Queue all configurations in a directory
   for config in experiment_configs/*.yml; do
     python aws_train_manager.py queue --config "$config" --mode train
   done
   ```

4. **Process jobs one by one**:
   ```bash
   # Process jobs one at a time
   python aws_train_manager.py process
   ```

5. **Monitor job status**:
   ```bash
   # Check AWS status
   python aws_deploy.py status
   
   # Check job queue status
   python aws_train_manager.py list-jobs
   ```

6. **Retrieve results when jobs are complete**:
   ```bash
   # Retrieve results for a specific configuration
   python aws_deploy.py retrieve --config experiment_configs/nafnet_LR_0.0001_BATCH_SIZE_4.yml
   ```

## Creating Custom Configuration Variations

You can create custom variations using the `create-variations` command with different parameter combinations:

```bash
# Vary learning rate and batch size
python aws_train_manager.py create-variations --base-config ThermalDenoising/conf_gopro_nafnet_rgb.yml --params "LR=0.0001,0.0002" "BATCH_SIZE=4,8"

# Vary model architecture
python aws_train_manager.py create-variations --base-config ThermalDenoising/conf_gopro_nafnet_rgb.yml --params "MODEL_CHANNELS=16,32,64" "NUM_RESBLOCKS=1,2,3"

# Vary training parameters
python aws_train_manager.py create-variations --base-config ThermalDenoising/conf_gopro_nafnet_rgb.yml --params "EMA_EVERY=50,100,200" "VALIDATE_EVERY=500,1000"
```

## Extending the System

The system is designed to be extensible:

1. **Custom Models**: Modify the model architecture parameters in the YAML configurations.
2. **Multi-Job Processing**: Update the `aws_train_manager.py` script to process multiple jobs simultaneously.
3. **Enhanced Monitoring**: Add additional monitoring capabilities through the AWS server.

## SSH Key Setup and Troubleshooting

We've included tools to help troubleshoot and fix SSH key issues that are common when working with AWS:

1. **Fix SSH Key Script**: 
   ```bash
   ./fix_ssh_key.sh
   ```
   This script:
   - Checks if your SSH key exists and has the correct permissions
   - Creates a copy of your key without spaces in the filename (which can cause connection issues)
   - Updates the aws_deploy.py script to use the fixed key
   - Tests the SSH connection

2. **Test SSH Connection**:
   ```bash
   ./aws_manage.sh test-ssh
   ```
   This command tests the SSH connection using the current key path in aws_deploy.py.

3. **Common SSH Issues**:
   - **Key permissions**: SSH keys must have 600 permissions (`chmod 600 ~/.ssh/your_key.pem`)
   - **Spaces in filenames**: SSH has trouble with spaces in key filenames
   - **AWS Security Groups**: Ensure your IP address is allowed in the AWS security group
   - **Instance Status**: Verify your AWS instance is running

## General Troubleshooting

1. **SSH Connection Issues**:
   - Run `./fix_ssh_key.sh` to automatically fix common SSH key problems
   - Check if the AWS server is reachable: `ping 3.140.193.132`

2. **Job Execution Failures**:
   - Check AWS logs: `ssh -i ~/.ssh/ThomasKiddDeepLearning.pem ubuntu@3.140.193.132 "tail -n 100 ~/Deep-Learning-Project/ThermalDenoising/logs/*.log"`
   - Ensure all required packages are installed on AWS with `./aws_deploy.py setup`

3. **File Transfer Issues**:
   - Manually transfer a file to test: `scp -i ~/.ssh/ThomasKiddDeepLearning.pem file.txt ubuntu@3.140.193.132:~/`
   - Verify file permissions in both local and remote directories

4. **Queue Processing Problems**:
   - Check `aws_training_status.json` for job status details
   - Reset a failed job status to "queued" to retry
