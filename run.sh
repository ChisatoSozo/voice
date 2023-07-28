#!/bin/bash

# Function to check if Python is installed
check_python_installed() {
    if command -v python3 &>/dev/null; then
        echo "Python is installed."
        echo "Running run.py..."
        python3 run.py
    elif command -v python &>/dev/null; then
        echo "Python is installed."
        echo "Running run.py..."
        python run.py
    else
        echo "Python is not installed on this system."
        echo "Please install Python to run this script."
    fi
}

# Call the function
check_python_installed