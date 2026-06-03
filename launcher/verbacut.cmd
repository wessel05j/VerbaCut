@echo off
setlocal
call "%~dp0run.bat" %*
exit /b %ERRORLEVEL%
