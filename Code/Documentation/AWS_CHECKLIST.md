# AWS Training System Setup Checklist

Follow these steps to ensure your AWS training system is working properly:

## Step 1: Fix SSH Key Issues

First, let's fix any SSH key issues:

```bash
# Run the quick fix script to fix SSH key issues
./quick_fix_ssh.sh
```

If the quick fix doesn't resolve the issue, try the more comprehensive troubleshooting:

```bash
# Run the detailed troubleshooting script
./fix_ssh_key.sh
```

## Step 2: Verify AWS Connectivity

Test if you can connect to the AWS server:

```bash
# Test SSH connection
./aws_manage.sh test-ssh
```

## Step 3: Set Up the AWS Environment

Once the SSH connection is working, set up the AWS environment:

```bash
# Set up AWS environment
./aws_deploy.py setup
```

## Step 4: Create Test Configurations

Generate some test configurations to experiment with:

```bash
# Create NAFNET experiment configurations
./create_nafnet_experiment.py
```

## Step 5: Queue and Run Jobs

Queue and run a test job:

```bash
# Queue a job
./aws_train_manager.py queue --config experiment_configs/nafnet_base.yml

# Process the job
./aws_train_manager.py process
```

## Step 6: Monitor Job Status

Check the status of running jobs:

```bash
# Check AWS status
./aws_deploy.py status

# List jobs in queue
./aws_train_manager.py list-jobs
```

## Step 7: Retrieve Results

After the job completes, retrieve the results:

```bash
# Retrieve results
./aws_deploy.py retrieve --config experiment_configs/nafnet_base.yml
```

## Troubleshooting Common Issues

1. **SSH Connection Issues**:
   - Key file path or format is incorrect
   - Key has improper permissions
   - AWS instance is not running or not accessible

2. **Job Execution Failures**:
   - Missing dependencies on AWS server
   - Incorrect configuration paths
   - AWS instance doesn't have enough resources

3. **Result Retrieval Problems**:
   - Job didn't complete successfully
   - Result directory structure is different than expected

For more detailed information, refer to `AWS_TRAINING_README.md`.
