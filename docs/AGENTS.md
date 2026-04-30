# AGENTS.md - Auto-Retrain System

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run system (one cycle)
python main.py --mode once --data data/train.csv --target target_column
```

## Project Structure

- `main.py` - Main orchestrator entry point
- `src/*.py` - Core modules: data_loader, evaluator, trainer, model_selector, deployer, monitor
- `config.yaml` - Configuration (paths, thresholds, model params)
- `colab_notebook.ipynb` - Ready-to-run Google Colab notebook

## Key Commands

```bash
# Generate sample data
python generate_data.py

# Run scheduled mode
python main.py --mode scheduled --data data/train.csv --target target_column

# In Google Colab: Runtime → Schedule fraction of code for automatic runs
```

## Configuration

Edit `config.yaml` to change:
- `TASK_TYPE`: "regression" or "classification"
- `MODELS.xgboost.n_trials`: Number of Optuna trials (reduce for Colab)
- `TRIGGERS.mse_threshold`: 0.05 = 5% tolerance for retraining

## Testing

```python
# Quick test
from main import AutoRetrainSystem
system = AutoRetrainSystem('config.yaml')
results = system.run_cycle()  # Uses data from config.yaml
```

## Colab Notes

- Reduce `n_trials` to 10-20 for faster execution
- Save checkpoints to Google Drive
- Use Runtime → Schedule fraction of code for automated runs
- Timeout: Colab disconnects after 90 min inactivity / 12 hr max session