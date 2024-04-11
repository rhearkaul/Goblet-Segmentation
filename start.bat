@echo off
echo Starting the application...

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install dependencies (if not already installed)
pip install -r requirements.txt

REM Run the GUI.py file and keep the window open
start cmd /k "python src\GUI.py & pause"

REM Deactivate the virtual environment
deactivate
