@echo off
:: Setup script for the semantic calculator MCP on Windows

echo Setting up the semantic calculator MCP...

:: Check if UV is installed
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo UV is not installed. Installing UV...
    pip install uv
    
    where uv >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install UV. Please install it manually with: pip install uv
        exit /b 1
    )
)

echo Creating virtual environment...
uv venv

if not exist ".venv" (
    echo Failed to create virtual environment.
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing dependencies...
uv pip install numpy scikit-learn sentence-transformers torch matplotlib umap-learn plotly pytest seaborn

echo Setup complete!
echo.
echo To use the semantic calculator, first activate the virtual environment:
echo   .venv\Scripts\activate
echo.
echo Then run one of the example scripts:
echo   python examples\emoji_3d_visualization.py
echo   python examples\visualize_emojikey.py
echo   python examples\dimension_analysis.py
echo.
echo To deactivate the virtual environment when you're done, run:
echo   deactivate
