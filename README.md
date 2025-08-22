# ML Project Setup

## Virtual Environment Setup

### 1. Activate the virtual environment:
```bash
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Verify installation:
```bash
python -c "import xgboost; import pandas; import sklearn; print('All packages installed successfully!')"
```

## Project Structure
```
ml/
├── venv/               # Virtual environment (git-ignored)
├── spikehistory/       # Raw CSV data files
├── processed/          # Processed data outputs
├── models/             # Saved model files
├── config.py           # Configuration settings
├── main.py             # Main execution script
├── parser.py           # Data parsing utilities
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Deactivating the Environment
When you're done working, deactivate the virtual environment:
```bash
deactivate
```

## Updating Dependencies
To add new packages:
```bash
pip install package-name
pip freeze > requirements.txt
```