#!/bin/bash

# Example workflow for running experiments on AWS
# This script demonstrates how to set up, create configurations, queue jobs, and process them

echo "=== AWS Training Workflow Example ==="
echo ""

# Step 1: Set up the AWS environment
echo "Step 1: Setting up AWS environment..."
python aws_deploy.py setup
echo ""

# Step 2: Create experiment configurations
echo "Step 2: Creating experiment configurations..."
echo "Creating NAFNet experiment variations..."
python create_nafnet_experiment.py

# Also create some custom variations
echo "Creating custom architecture variations..."
python aws_train_manager.py create-variations --base-config ThermalDenoising/conf_gopro_nafnet_rgb.yml --params "MODEL_CHANNELS=16,64" "CHANNEL_MULT=[1,2,2,2],[1,2,4,8]"
echo ""

# Step 3: Queue jobs for training
echo "Step 3: Queueing jobs for training..."
echo "Queueing experiment configurations..."

# Queue a subset of created configurations
echo "Queueing base configuration..."
python aws_train_manager.py queue --config experiment_configs/nafnet_base.yml --mode train

echo "Queueing learning rate variations..."
python aws_train_manager.py queue --config experiment_configs/nafnet_LR_0.00005_BATCH_SIZE_4.yml --mode train
python aws_train_manager.py queue --config experiment_configs/nafnet_LR_0.0002_BATCH_SIZE_4.yml --mode train
echo ""

# Step 4: Process the jobs
echo "Step 4: Processing jobs..."
echo "Starting the first job..."
python aws_train_manager.py process

echo "Checking status..."
python aws_deploy.py status
echo ""

echo "To continue processing jobs, run:"
echo "  python aws_train_manager.py process"
echo ""

echo "To check AWS status, run:"
echo "  python aws_deploy.py status"
echo ""

echo "To check job queue status, run:"
echo "  python aws_train_manager.py list-jobs"
echo ""

echo "When jobs complete, retrieve results with:"
echo "  python aws_deploy.py retrieve --config experiment_configs/nafnet_base.yml"
echo ""

echo "=== Workflow Example Complete ==="
echo "Follow the AWS_TRAINING_README.md for more detailed instructions."
