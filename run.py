import os
import subprocess

# Create a virtual environment
subprocess.call(['python', '-m', 'venv', 'venv'])

# Activate the virtual environment
if os.name == 'nt':  # For Windows
    venv_activate = os.path.join('venv', 'Scripts', 'activate')
else:  # For Unix/Linux/macOS
    venv_activate = os.path.join('venv', 'bin', 'activate')

subprocess.call(['source', venv_activate], shell=True)

# Install dependencies
subprocess.call(['pip', 'install', '-r', 'requirements.txt'])

# Install the project package
subprocess.call(['pip', 'install', '.'])

# Run the project
subprocess.call(['goblet-segmentation'])
