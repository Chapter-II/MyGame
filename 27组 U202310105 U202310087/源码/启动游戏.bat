@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if exist "local.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("local.env") do (
    if not "%%~A"=="" set "%%~A=%%~B"
  )
)

if defined DEEPSEEK_API_KEY (
  echo [mygame] DeepSeek API key loaded
) else (
  echo [mygame] DEEPSEEK_API_KEY not set - offline text commands only
)

start "" ".venv\Scripts\pythonw.exe" -m mygame
endlocal
