#!/bin/bash

# One-step script to fix SSH connection issue with spaces in key filename

echo "=== Quick Fix for SSH Connection Issues ==="
echo ""

# Check if the proper key exists - first check if we already have the fixed key
fixed_key="$HOME/.ssh/ThomasKiddDeepLearning.pem"
original_key="$HOME/.ssh/Thomas Kidd Deep Learning.pem"

# Check if fixed key already exists
if [ -f "$fixed_key" ]; then
    echo "Fixed SSH key already exists at: $fixed_key"
    chmod 600 "$fixed_key"
    # Update aws_deploy.py to use the fixed key
    sed -i 's|AWS_KEY_PATH = ".*"|AWS_KEY_PATH = "~/.ssh/ThomasKiddDeepLearning.pem"|' aws_deploy.py
elif [ -f "$original_key" ]; then
    echo "Found original key with spaces. Setting permissions and creating copy..."
    chmod 600 "$original_key"
    cp "$original_key" "$fixed_key"
    chmod 600 "$fixed_key"
    
    # Update the aws_deploy.py file to use a simple key path without spaces
    sed -i 's|AWS_KEY_PATH = ".*"|AWS_KEY_PATH = "~/.ssh/ThomasKiddDeepLearning.pem"|' aws_deploy.py
    echo "Done! SSH key has been fixed."
else
    # Neither key exists, check alternative locations
    echo "SSH key not found at expected locations."
    echo "Searching for other .pem files in your ~/.ssh directory..."
    
    # Find any .pem files in ~/.ssh
    pem_files=$(find ~/.ssh -name "*.pem" 2>/dev/null)
    
    if [ -n "$pem_files" ]; then
        echo "Found the following .pem files:"
        echo "$pem_files"
        echo ""
        echo "Please update the AWS_KEY_PATH in aws_deploy.py to use one of these keys,"
        echo "or copy your AWS key to: ~/.ssh/ThomasKiddDeepLearning.pem"
    else
        echo "No .pem files found in ~/.ssh directory."
        echo "Please place your AWS SSH key at: ~/.ssh/ThomasKiddDeepLearning.pem"
        echo "and ensure it has 600 permissions (chmod 600 ~/.ssh/ThomasKiddDeepLearning.pem)"
    fi
fi

echo ""
echo "To verify everything is working, try running:"
echo "./aws_deploy.py setup"
