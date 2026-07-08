@echo off
cd /d "%~dp0"
python -m src.document_scanner.scan_interactive
pause
