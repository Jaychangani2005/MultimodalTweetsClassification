@echo off
echo ========================================
echo Multimodal Tweet Classifier - Streamlit
echo ========================================
echo.
echo Starting Streamlit server...
echo.
cd /d "%~dp0"
streamlit run informative_Attention_graph_sequence1.py
pause
