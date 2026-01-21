@echo off
echo Starting Multimodal Tweet Classification App...
echo.

REM Check if we're in the right directory
if not exist "informative_Attention_graph.py" (
    echo Error: Please run this script from the frontend directory
    echo Current directory should contain informative_Attention_graph.py
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Error: Streamlit is not installed
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Launch the Streamlit app
echo Launching Streamlit app...
echo.
echo The app will open in your default web browser
echo Press Ctrl+C to stop the app
echo.

streamlit run informative_Attention_graph.py

pause