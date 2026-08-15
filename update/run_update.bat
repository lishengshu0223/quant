@echo off
chcp 65001 >nul
echo ================================================
echo         Quant Data Update Script
echo ================================================
echo.

set PROJECT_DIR=F:\quant
set PYTHON_EXE=E:\Anaconda\envs\multifactor\python.exe
set RQSDK_PATH=E:\Anaconda\envs\multifactor\Scripts\rqsdk.exe
set BUNDLE_DIR=F:\Trade_data\rq_backtest_data
set LOG_DIR=F:\Trade_data\logs

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f "usebackq" %%d in (`powershell -Command "Get-Date -Format 'yyyyMMdd'"`) do (
    set LOG_FILE=%LOG_DIR%\download_%%d.log
)

cd /d %PROJECT_DIR%

echo Project Dir: %PROJECT_DIR%
echo Python Path: %PYTHON_EXE%
echo RQSDK Path: %RQSDK_PATH%
echo Bundle Dir: %BUNDLE_DIR%
echo Log File: %LOG_FILE%
echo Start Time: %date% %time%
echo [%date% %time%] Start Time: %date% %time% >> "%LOG_FILE%"
echo.

echo --- Step 1: Update Latest Trading Day ---
echo [%date% %time%] --- Step 1: Update Latest Trading Day --- >> "%LOG_FILE%"
%PYTHON_EXE% %PROJECT_DIR%\update\update_latest.py
echo [%date% %time%] Step 1 completed >> "%LOG_FILE%"
echo.

echo --- Step 2: Update Missing History Data ---
echo [%date% %time%] --- Step 2: Update Missing History Data --- >> "%LOG_FILE%"
%PYTHON_EXE% %PROJECT_DIR%\update\daily_update.py --mode history
echo [%date% %time%] Step 2 completed >> "%LOG_FILE%"
echo.

echo --- Step 3: Update Backtest Data (Base) ---
echo [%date% %time%] --- Step 3: Update Backtest Data (Base) --- >> "%LOG_FILE%"
"%RQSDK_PATH%" update-data -d "%BUNDLE_DIR%" --base
echo [%date% %time%] Step 3 completed >> "%LOG_FILE%"
echo.

echo --- Step 4: Update Backtest Data (Stock) ---
echo [%date% %time%] --- Step 4: Update Backtest Data (Stock) --- >> "%LOG_FILE%"
echo This step may take a long time, please wait...
"%RQSDK_PATH%" update-data -d "%BUNDLE_DIR%" --minbar stock -c 4
echo [%date% %time%] Step 4 completed >> "%LOG_FILE%"
echo.

echo --- Step 5: Update Backtest Data (Futures) ---
echo [%date% %time%] --- Step 5: Update Backtest Data (Futures) --- >> "%LOG_FILE%"
echo This step may take a long time, please wait...
"%RQSDK_PATH%" update-data -d "%BUNDLE_DIR%" --minbar futures -c 4
echo [%date% %time%] Step 5 completed >> "%LOG_FILE%"
echo.

echo End Time: %date% %time%
echo [%date% %time%] End Time: %date% %time% >> "%LOG_FILE%"
echo ================================================
echo                  Update Completed
echo ================================================
echo [%date% %time%] Update Completed >> "%LOG_FILE%"