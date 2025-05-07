#!/bin/bash

# Usage helper function
usage() {
  echo "AWS Training Management Script"
  echo "Usage: $0 [command]"
  echo ""
  echo "Commands:"
  echo "  setup                Set up the AWS environment"
  echo "  test-ssh             Test SSH connection to AWS server"
  echo "  create-exp           Create NAFNet experiment configurations"
  echo "  custom-var BASE P1 P2...  Create custom variations (e.g., custom-var ThermalDenoising/conf_gopro_nafnet_rgb.yml \"LR=0.0001,0.0002\" \"BATCH_SIZE=4,8\")"
  echo "  queue-all DIR        Queue all configs in directory for training"
  echo "  queue CONFIG         Queue a specific config for training"
  echo "  process              Process the next job in the queue"
  echo "  list-configs         List all available configurations"
  echo "  list-jobs            List all jobs in the queue"
  echo "  status               Check AWS status"
  echo "  retrieve CONFIG      Retrieve results for a config"
  echo "  help                 Show this help message"
  echo ""
  exit 1
}

# Check if command is provided
if [ "$#" -lt 1 ]; then
  usage
fi

# Process commands
command="$1"
shift

case "$command" in
  setup)
    python aws_deploy.py setup
    ;;
  
  test-ssh)
    echo "Testing SSH connection to AWS server..."
    ssh_key_path=$(grep "AWS_KEY_PATH" aws_deploy.py | sed -n 's/AWS_KEY_PATH = "\(.*\)".*/\1/p' | sed 's/\\\\/\\/g')
    server=$(grep "AWS_SERVER" aws_deploy.py | sed -n 's/AWS_SERVER = "\(.*\)".*/\1/p')
    
    # Extract actual path without escaping
    key_path=$(echo "$ssh_key_path" | sed 's/\\\\/\\/g')
    
    # Test SSH connection
    echo "Using key: $key_path"
    echo "Connecting to: $server"
    echo "Command: ssh -i '$key_path' $server 'echo Connection successful'"
    ssh -i "$key_path" $server "echo Connection successful"
    
    if [ $? -eq 0 ]; then
      echo "SSH connection successful!"
    else
      echo "SSH connection failed. Check your SSH key and server details."
    fi
    ;;
    
  create-exp)
    python create_nafnet_experiment.py
    ;;
    
  custom-var)
    # Need at least 2 more arguments: base config and at least one parameter variation
    if [ "$#" -lt 2 ]; then
      echo "Error: custom-var requires a base config and at least one parameter variation"
      usage
    fi
    
    base_config="$1"
    shift
    params=("$@")
    
    # Construct the command
    cmd="python aws_train_manager.py create-variations --base-config $base_config"
    for param in "${params[@]}"; do
      cmd="$cmd --params \"$param\""
    done
    
    # Execute the command
    eval "$cmd"
    ;;
    
  queue-all)
    # Need directory argument
    if [ "$#" -lt 1 ]; then
      echo "Error: queue-all requires a directory argument"
      usage
    fi
    
    dir="$1"
    # Check if directory exists
    if [ ! -d "$dir" ]; then
      echo "Error: Directory $dir does not exist"
      exit 1
    fi
    
    echo "Queueing all configurations in $dir for training..."
    count=0
    for config in "$dir"/*.yml; do
      if [ -f "$config" ]; then
        echo "  Queueing $config..."
        python aws_train_manager.py queue --config "$config" --mode train
        count=$((count+1))
      fi
    done
    
    echo "Queued $count configurations"
    ;;
    
  queue)
    # Need config argument
    if [ "$#" -lt 1 ]; then
      echo "Error: queue requires a config argument"
      usage
    fi
    
    config="$1"
    # Check if file exists
    if [ ! -f "$config" ]; then
      echo "Error: Config file $config does not exist"
      exit 1
    fi
    
    # Queue the job
    python aws_train_manager.py queue --config "$config" --mode train
    ;;
    
  process)
    python aws_train_manager.py process
    ;;
    
  list-configs)
    python aws_train_manager.py list-configs
    ;;
    
  list-jobs)
    python aws_train_manager.py list-jobs
    ;;
    
  status)
    python aws_deploy.py status
    ;;
    
  retrieve)
    # Need config argument
    if [ "$#" -lt 1 ]; then
      echo "Error: retrieve requires a config argument"
      usage
    fi
    
    config="$1"
    # Check if file exists
    if [ ! -f "$config" ]; then
      echo "Error: Config file $config does not exist"
      exit 1
    fi
    
    # Retrieve results
    python aws_deploy.py retrieve --config "$config"
    ;;
    
  help)
    usage
    ;;
    
  *)
    echo "Unknown command: $command"
    usage
    ;;
esac
