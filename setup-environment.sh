#!/bin/bash

# Get the absolute path of the current directory
DIR=$(pwd)

# Extract just the directory name
DIR_NAME=${DIR##*/}

# Create virtualenv with directory name
python3 -m venv .$DIR_NAME

# Activate the virtualenv
source .$DIR_NAME/bin/activate

# Check if virtualenv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtualenv not activated. Exiting."
    exit 1
fi

# Install packages
pip install -r requirements.txt

echo "Environment '$DIR_NAME' setup complete"