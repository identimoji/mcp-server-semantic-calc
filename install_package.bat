@echo off
REM Script to install the semantic calculator package

REM Check if a virtual environment exists
if not exist ".venv" (
  echo Creating virtual environment...
  python -m venv .venv
)

REM Activate the virtual environment
call .venv\Scripts\activate

REM Install the package in development mode
echo Installing package in development mode...
pip install -e .

echo Installation complete!
echo To use the package, activate the virtual environment with:
echo   .venv\Scripts\activate
echo.
echo Then you can import the package in Python:
echo   from semantic_calculator.core import SemanticCalculator
echo.
echo Or run one of the example scripts:
echo   python examples\vector_operations.py
