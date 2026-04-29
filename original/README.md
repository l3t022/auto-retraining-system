# Auto-Retrain System

Sistema de auto-entrenamiento de modelos ML con selección automática basada en métricas.

## Características

- **Monitoreo de datos**: Detecta automáticamente cuando hay nuevos datos
- **Evaluación continua**: Compara métricas actuales vs baseline
- **Auto-reentrenamiento**: Re-entrena cuando las métricas bajan del threshold
- **Búsqueda de hiperparámetros**: Usa Optuna (Bayesian optimization)
- **Selección automática**: Elige el mejor modelo entre candidatos
- **Versionado**: Mantiene historial de modelos y métricas

## Instalación

```bash
# Clonar o copiar el proyecto
cd auto_retrain_system

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Para Google Colab:
!pip install -r requirements.txt
```

## Configuración

Editar `config.yaml`:

```yaml
# Rutas
DATA_PATH: "data/train.csv"
MODEL_PATH: "models/"
LOGS_PATH: "logs/"

# Tipo de tarea
TASK_TYPE: "regression"  # o "classification"

# Modelos a usar
MODELS:
  xgboost:
    enabled: true
    n_trials: 20
    timeout: 300

# Thresholds de re-entrenamiento
TRIGGERS:
  mse_threshold: 0.05    # 5% tolerancia
  accuracy_threshold: 0.03
```

## Uso Rápido

### En Python

```python
from main import AutoRetrainSystem

# Inicializar sistema
system = AutoRetrainSystem('config.yaml')

# Ejecutar un ciclo
results = system.run_cycle()

print(results)

# O con datos específicos
import pandas as pd
df = pd.read_csv('data/train.csv')
X = df.drop(columns=['target'])
y = df['target']

results = system.run_cycle(X, y)
```

### Desde línea de comandos

```bash
# Una ejecución
python main.py --mode once --data data/train.csv --target target_column

# Modo programado
python main.py --mode scheduled --data data/train.csv --target target_column
```

### En Google Colab

```python
# Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# Importar y ejecutar
import sys
sys.path.append('/content/auto_retrain_system')

from main import AutoRetrainSystem

system = AutoRetrainSystem('/content/auto_retrain_system/config.yaml')
results = system.run_cycle()
```

## Estructura del Proyecto

```
auto_retrain_system/
├── config.yaml           # Configuración
├── requirements.txt      # Dependencias
├── main.py              # Orquestador principal
├── plan.md              # Plan de implementación
├── src/
│   ├── __init__.py
│   ├── data_loader.py   # Carga de datos
│   ├── evaluator.py     # Evaluación de modelos
│   ├── trainer.py       # Entrenamiento con Optuna
│   ├── model_selector.py # Selección de mejor modelo
│   ├── deployer.py      # Despliegue de modelos
│   └── monitor.py       # Monitoreo de datos
├── models/              # Modelos guardados
├── logs/                # Métricas e historial
└── data/                # Datos de entrenamiento
```

## Funcionamiento

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 1. Datos   │───▶│ 2. Evalúa   │───▶│ 3. Check    │
│ Nuevos     │    │ Modelo      │    │ Metrics    │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                              metrics.drop?  │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 6. Deploy  │◀───│ 5. Selecc.  │◀───│ 4. Optimiza │
│ Mejor      │    │ Mejor       │    │ HP (Optuna) │
│ Modelo     │    │ vs Actual   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Monitoreo Programado

### Opción 1: Schedule (Python)

```python
from src.monitor import ScheduledMonitor

monitor = ScheduledMonitor()
monitor.schedule_hours(24, lambda: system.run_cycle())
monitor.run_continuously()
```

### Opción 2: Google Colab Scheduler

1. Ir a **Runtime → Schedule fraction of code**
2. Configurar frecuencia (ej: daily)
3. El código se ejecutará automáticamente

### Opción 3: GitHub Actions

Crear `.github/workflows/train.yml`:
```yaml
name: Auto-Train
on:
  schedule:
    - cron: '0 9 * * *'  # Daily at 9 AM
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run training
        run: python main.py --mode once
```

## Métricas Monitoreadas

### Regresión
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (R-squared)

### Clasificación
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## Threshold Default

- **Regresión**: Re-entrena si MSE aumenta >5%
- **Clasificación**: Re-entrena si accuracy baja >3%

## Tech Stack

- **Optuna**: HPO (Bayesian optimization with TPE)
- **XGBoost**: Gradient boosting
- **LightGBM**: Light gradient boosting
- **scikit-learn**: Utilities

## Notas

- Requiere Python 3.10+
- Optimizado para Google Colab Free
- Compatible con CSV y Google Drive

## Licencia

MIT