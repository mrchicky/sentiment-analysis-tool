@echo off
cd /d %~dp0
call .venv\Scripts\activate
streamlit run src\sentiment_analyzer\app.py
pause