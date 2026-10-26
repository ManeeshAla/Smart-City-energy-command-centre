@echo off
REM Smart City Energy Command Center - one-click launcher
cd /d "%~dp0"

echo ============================================================
echo  Smart City Energy - build pipeline then launch dashboard
echo ============================================================

REM Optional: set your MongoDB Atlas connection string here (or leave blank
REM to use the local NoSQL fallback). Never commit a real password.
REM set MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/

if not exist "smart_city_energy_timeseries.csv" (
    echo [1/2] Generating hourly dataset ...
    python generate_data.py || goto :err
)

if not exist "artifacts\models.joblib" (
    echo [2/2] Training models and building batch layer ...
    python pipeline.py || goto :err
)

echo.
echo Starting dashboard at http://127.0.0.1:8050  (press Ctrl+C to stop)
start "" http://127.0.0.1:8050
python dashboard.py
goto :eof

:err
echo.
echo Something failed above. Make sure dependencies are installed:
echo    python -m pip install -r requirements.txt
pause
