import os
import subprocess

# Create a virtual environment if it doesn't exist
if not os.path.exists('venv'):
    subprocess.call(['python', '-m', 'venv', 'venv'])

# Activate the virtual environment
if os.name == 'nt':  # For Windows
    venv_activate = os.path.join('venv', 'Scripts', 'activate')
else:  # For Unix/Linux/macOS
    venv_activate = os.path.join('venv', 'bin', 'activate')

subprocess.call(['source', venv_activate], shell=True)

# Check if dependencies are already installed
if not os.path.exists('venv/installed.txt'):
    # Install dependencies
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])

    # Create a file to mark that dependencies are installed
    open('venv/installed.txt', 'a').close()

# Install the project package
subprocess.call(['pip', 'install', '.'])

# Run the project
subprocess.call(['goblet-segmentation'])

