@echo off
REM Batch file to run the Roulette Prediction System on Windows

IF "%1"=="" (
    python run.py
    exit /b
)

IF "%1"=="train" (
    python run.py train %2 %3 %4 %5 %6 %7 %8 %9
    exit /b
)

IF "%1"=="test" (
    python run.py test %2 %3 %4 %5 %6 %7 %8 %9
    exit /b
)

IF "%1"=="predict" (
    python run.py predict %2 %3 %4 %5 %6 %7 %8 %9
    exit /b
)

IF "%1"=="camera" (
    python run.py camera %2 %3 %4 %5 %6 %7 %8 %9
    exit /b
)

IF "%1"=="simulation" (
    python run.py simulation %2 %3 %4 %5 %6 %7 %8 %9
    exit /b
)

echo Unknown command: %1
python run.py 