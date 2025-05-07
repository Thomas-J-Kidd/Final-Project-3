#!/bin/bash

# Script to fix SSH key issues with the AWS training system

echo "=== SSH Key Troubleshooting Script ==="
echo ""

# Path to the original SSH key with spaces
# Check if the proper key exists - first check if we already have the fixed key
fixed_key="$HOME/.ssh/ThomasKiddDeepLearning.pem"
original_key="$HOME/.ssh/Thomas Kidd Deep Learning.pem"

echo "=== SSH Key Detection ==="
echo "Checking for SSH keys..."

# Function to check permissions and fix if needed
check_and_fix_permissions() {
    local key_path=$1
    local perms=$(stat -c %a "$key_path" 2>/dev/null)
    
    if [ "$perms" != "600" ]; then
        echo "Setting correct permissions (600) on key: $key_path"
        chmod 600 "$key_path"
    else
        echo "Key permissions are correct (600) for: $key_path"
    fi
}

# Check if fixed key already exists
if [ -f "$fixed_key" ]; then
    echo "Found SSH key without spaces: $fixed_key"
    check_and_fix_permissions "$fixed_key"
    
    echo "Updating aws_deploy.py to use the fixed key path..."
    sed -i "s|AWS_KEY_PATH = \".*\"|AWS_KEY_PATH = \"$fixed_key\"|" aws_deploy.py
    
elif [ -f "$original_key" ]; then
    echo "Found SSH key with spaces in filename: $original_key"
    check_and_fix_permissions "$original_key"
    
    # Copy the key to a new filename without spaces
    echo "Creating a copy of the key without spaces in the filename..."
    cp "$original_key" "$fixed_key"
    chmod 600 "$fixed_key"
    
    # Update the aws_deploy.py file to use the fixed key
    echo "Updating aws_deploy.py to use the fixed key path..."
    sed -i "s|AWS_KEY_PATH = \".*\"|AWS_KEY_PATH = \"$fixed_key\"|" aws_deploy.py
    echo "Created fixed key at: $fixed_key"
    
else
    # Neither key exists, search for alternatives
    echo "SSH key not found at expected locations."
    echo "Searching for other .pem files in your ~/.ssh directory..."
    
    # Find any .pem files in ~/.ssh
    pem_files=$(find ~/.ssh -name "*.pem" 2>/dev/null)
    
    if [ -n "$pem_files" ]; then
        echo "Found the following .pem files:"
        echo "$pem_files"
        echo ""
        echo "Please choose one of these keys and update aws_deploy.py manually,"
        echo "or copy your AWS key to: $fixed_key"
        
        # Offer to use the first found key
        first_key=$(echo "$pem_files" | head -n 1)
        if [ -n "$first_key" ]; then
            read -p "Would you like to use $first_key as your AWS key? (y/n): " use_first
            if [[ "$use_first" =~ ^[Yy] ]]; then
                cp "$first_key" "$fixed_key"
                chmod 600 "$fixed_key"
                sed -i "s|AWS_KEY_PATH = \".*\"|AWS_KEY_PATH = \"$fixed_key\"|" aws_deploy.py
                echo "Copied $first_key to $fixed_key and updated aws_deploy.py"
            fi
        fi
    else
        echo "No .pem files found in ~/.ssh directory."
        echo "Please place your AWS SSH key at: $fixed_key"
        echo "and ensure it has 600 permissions (chmod 600 $fixed_key)"
    fi
fi

echo ""
echo "=== SSH Connection Test ==="

# Get the server from aws_deploy.py
server=$(grep "AWS_SERVER" aws_deploy.py | sed -n 's/AWS_SERVER = "\(.*\)".*/\1/p')

# Try to connect with the fixed key if it exists
if [ -f "$fixed_key" ]; then
    echo "Testing connection to $server using key $fixed_key..."
    echo "Command: ssh -i \"$fixed_key\" $server \"echo SSH connection successful\""
    
    # Try SSH with a timeout to avoid hanging if server is unreachable
    timeout 10 ssh -i "$fixed_key" $server "echo SSH connection successful"
    
    if [ $? -eq 0 ]; then
        echo "✅ SSH connection successful!"
    elif [ $? -eq 124 ]; then
        echo "❌ SSH connection timed out. Please check:"
        echo "   1. Your AWS instance is running"
        echo "   2. The IP address/hostname is correct"
        echo "   3. Network connectivity to the server"
    else
        echo "❌ SSH connection failed. Please check:"
        echo "   1. Your AWS instance is running"
        echo "   2. Your IP address is allowed in the AWS security group"
        echo "   3. Your SSH key is correct for this instance"
    fi
    # Additional diagnostics
    echo ""
    echo "=== Diagnostics ==="
    echo "Checking host connectivity..."
    host_part=$(echo "$server" | cut -d'@' -f2)
    ping -c 1 -W 2 "$host_part" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Host $host_part is reachable"
    else
        echo "❌ Cannot reach host $host_part"
        echo "   * Check if the hostname/IP is correct"
        echo "   * Check your internet connection"
    fi
else
    echo "No SSH key available for testing connection"

fi

echo ""
echo "=== Troubleshooting Tips ==="
echo "1. If you still have issues, try running: ./aws_manage.sh test-ssh"
echo "2. Make sure your AWS EC2 instance is running"
echo "3. Check AWS security group allows SSH (port 22) from your IP"
echo "4. Try copying your key manually: cp /path/to/your/key.pem ~/.ssh/ThomasKiddDeepLearning.pem"
echo "5. If using VPN, try disconnecting as it might block port 22"
echo ""
echo "=== Next Steps ==="
echo "If your SSH connection is working, run:"
echo "./aws_deploy.py setup"
