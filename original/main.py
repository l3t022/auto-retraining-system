"""
Main Orchestrator
Orquestador principal del sistema de auto-retraining.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.evaluator import ModelEvaluator
from src.trainer import ModelTrainer
from src.model_selector import ModelSelector
from src.deployer import ModelDeployer
from src.monitor import DataMonitor, ScheduledMonitor


class AutoRetrainSystem:
    """Sistema completo de auto-retraining."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Inicializar componentes
        self._init_components()
        
        self.logger.info("Sistema de auto-retraining inicializado")
    
    def _setup_logging(self):
        """Configura logging."""
        log_dir = self.config.get('LOGS_PATH', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"auto_retrain_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_components):
        """Inicializa los componentes del sistema."""
        paths = self.config.get('PATHS', {})
        
        # Data Loader
        self.data_loader = DataLoader(
            data_path=paths.get('DATA_PATH', 'data/train.csv'),
            baseline_path=paths.get('BASELINE_PATH', 'logs/baseline_data.json'),
        )
        
        # Evaluator
        triggers = self.config.get('TRIGGERS', {})
        self.evaluator = ModelEvaluator(
            model_path=paths.get('MODEL_PATH', 'models/current.pkl'),
            baseline_path=paths.get('BASELINE_PATH', 'logs/baseline_metrics.json'),
            metrics_history_path=paths.get('METRICS_HISTORY_PATH', 'logs/metrics_history.json'),
            task_type=self.config.get('TASK_TYPE', 'regression'),
            mse_threshold=triggers.get('mse_threshold', 0.05),
            accuracy_threshold=triggers.get('accuracy_threshold', 0.03),
        )
        
        # Deployer
        self.deployer = ModelDeployer(
            models_dir=paths.get('MODEL_PATH', 'models/').replace('/current.pkl', ''),
        )
        
        # Model Trainer
        training = self.config.get('TRAINING', {})
        models_config = self.config.get('MODELS', {})
        
        xgb_config = models_config.get('xgboost', {})
        self.trainer = ModelTrainer(
            task_type=self.config.get('TASK_TYPE', 'regression'),
            model_type='xgboost',
            n_trials=xgb_config.get('n_trials', 20),
            timeout=xgb_config.get('timeout', 300),
            cv_folds=training.get('cv_folds', 5),
            random_state=training.get('random_state', 42),
        )
        
        # Selector
        self.selector = ModelSelector(
            models_dir=paths.get('MODEL_PATH', 'models/').replace('/current.pkl', ''),
            task_type=self.config.get('TASK_TYPE', 'regression'),
        )
    
    def check_data(self) -> Tuple[bool, Dict]:
        """ Verifica si hay nuevos datos."""
        has_new, info = self.data_loader.get_new_data_info()
        
        if has_new:
            self.logger.info(f"Nuevos datos detectados: {info.get('new_rows', 0)} nuevas filas")
        else:
            self.logger.info("No hay nuevos datos")
        
        return has_new, info
    
    def evaluate_current_model(self, X, y) -> Dict:
        """Evalúa el modelo actual."""
        self.logger.info("Evaluando modelo actual...")
        
        try:
            metrics = self.evaluator.evaluate(X, y)
            self.logger.info(f"Métricas actuales: {metrics}")
            return metrics
        except Exception as e:
            self.logger.warning(f"Error evaluando modelo: {e}")
            return {}
    
    def should_retrain(self, metrics: Dict) -> Tuple[bool, str]:
        """Determina si debe re-entrenar."""
        should, reason = self.evaluator.should_retrain(metrics)
        
        if should:
            self.logger.info(f"Re-entrenamiento necesario: {reason}")
        else:
            self.logger.info(f"Sin re-entrenar: {reason}")
        
        return should, reason
    
    def optimize_model(self, X, y, X_val=None, y_val=None) -> Tuple:
        """Ejecuta optimización de hiperparámetros."""
        self.logger.info("Iniciando optimización de hiperparámetros...")
        
        try:
            model, params, score = self.trainer.train(X, y, X_val, y_val)
            
            self.logger.info(f"Mejor score: {score}")
            self.logger.info(f"Mejores parámetros: {params}")
            
            return model, params, score
        except Exception as e:
            self.logger.error(f"Error en optimización: {e}")
            raise
    
    def compare_and_deploy(
        self,
        new_model,
        new_metrics: Dict,
        current_metrics: Optional[Dict] = None,
    ) -> bool:
        """Compara y despliega el mejor modelo."""
        self.logger.info("Comparando modelos...")
        
        if current_metrics is None:
            current_metrics = self.evaluator.baseline_metrics
        
        # Usar selector para determinar si reemplazar
        should_replace, reason = self.selector.should_replace(
            current_model_path=self.deployer.get_current()['model_path'] if self.deployer.get_current() else "",
            current_metrics=current_metrics,
            new_model_path="temp_new_model",
            new_metrics=new_metrics,
            improvement_threshold=0.01,
        )
        
        if should_replace:
            self.logger.info(f"Desplegando nuevo modelo: {reason}")
            
            # Guardar modelo
            version = self.deployer.save_model(
                model=new_model,
                model_name="xgboost",
                metrics=new_metrics,
                metadata={'best_params': getattr(self.trainer, 'best_params', {})},
            )
            
            # Actualizar baseline
            self.evaluator.register_metrics(new_metrics, set_as_baseline=True)
            
            return True
        else:
            self.logger.info(f"No se despliega: {reason}")
            return False
    
    def train_initial_model(self, X, y) -> bool:
        """Entrena el modelo inicial (sin baseline)."""
        self.logger.info("Entrenando modelo inicial...")
        
        try:
            # Optimizar
            model, params, score = self.optimize_model(X, y)
            
            # Evaluar en training set para obtener métricas
            import numpy as np
            from sklearn.metrics import mean_squared_error
            
            y_pred = model.predict(X)
            metrics = {
                'mse': float(mean_squared_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            }
            
            # Deploy
            self.compare_and_deploy(model, metrics)
            
            # Guardar baseline de datos
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                df = X.copy()
                df['_target_'] = y
                self.data_loader.save_baseline(df)
            
            self.logger.info("Modelo inicial entrenado y guardado")
            return True
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo inicial: {e}")
            return False
    
    def run_cycle(self, X=None, y=None) -> Dict:
        """Ejecuta un ciclo completo del sistema."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'had_new_data': False,
            'retrained': False,
            'deployed': False,
            'details': {},
        }
        
        # Step 1: Cargar datos si no se proporcionan
        if X is None or y is None:
            try:
                df = self.data_loader.load_data()
                target_col = self.config.get('TARGET_COLUMN', None)
                if target_col is None:
                    # Usar última columna como target
                    target_col = df.columns[-1]
                
                X = df.drop(columns=[target_col])
                y = df[target_col]
                results['had_new_data'] = True
            except Exception as e:
                self.logger.error(f"Error cargando datos: {e}")
                results['details']['error'] = str(e)
                return results
        
        # Step 2: Verificar si hay modelo actual
        current_model = self.deployer.get_model()
        
        if current_model is None:
            self.logger.info("No hay modelo - entrenando inicial")
            self.train_initial_model(X, y)
            results['retrained'] = True
            results['deployed'] = True
            return results
        
        # Step 3: Evaluar modelo actual
        current_metrics = self.evaluate_current_model(X, y)
        
        if not current_metrics:
            # Error al evaluar - re-entrenar
            self.logger.warning("No se pudo evaluar modelo - entrenando nuevo")
            model, params, score = self.optimize_model(X, y)
            self.compare_and_deploy(model, {'score': score})
            results['retrained'] = True
            results['deployed'] = True
            return results
        
        # Step 4: Verificar si debe re-entrenar
        should_retrain, reason = self.should_retrain(current_metrics)
        
        results['details']['evaluation'] = current_metrics
        results['details']['should_retrain_reason'] = reason
        
        if should_retrain:
            # Step 5: Optimizar modelo
            new_model, params, score = self.optimize_model(X, y)
            
            # Step 6: Evaluar nuevo modelo
            import numpy as np
            from sklearn.metrics import mean_squared_error
            
            y_pred = new_model.predict(X)
            new_metrics = {
                'mse': float(mean_squared_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            }
            
            # Step 7: Comparar y deploy
            deployed = self.compare_and_deploy(new_model, new_metrics, current_metrics)
            
            results['retrained'] = True
            results['deployed'] = deployed
            results['details']['new_metrics'] = new_metrics
            results['details']['params'] = params
        else:
            self.logger.info("Modelo actual es bueno - sin cambios")
        
        return results
    
    def run_scheduled(self, data_path: str, target_col: str, n_executions: int = None):
        """Ejecuta el sistema de forma programada."""
        import pandas as pd
        
        schedule_cfg = self.config.get('SCHEDULE', {})
        
        if not schedule_cfg.get('enabled', False):
            self.logger.warning("Schedule deshabilitado")
            return
        
        frequency = schedule_cfg.get('frequency_hours', 24)
        
        self.logger.info(f"Iniciando scheduler cada {frequency} horas")
        
        execution_count = 0
        
        while True:
            try:
                # Cargar datos
                df = pd.read_csv(data_path)
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Ejecutar ciclo
                results = self.run_cycle(X, y)
                
                execution_count += 1
                self.logger.info(f"Ejecución {execution_count} completada: {results}")
                
                # Verificar si alcanzar límite de ejecuciones
                if n_executions and execution_count >= n_executions:
                    self.logger.info(f"Límite de ejecuciones alcanzado: {n_executions}")
                    break
                
            except Exception as e:
                self.logger.error(f"Error en ciclo programado: {e}")
            
            # Esperar siguiente ejecución
            import time
            time.sleep(frequency * 3600)


def main():
    """Punto de entrada principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-Retrain System')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--mode', choices=['once', 'scheduled'], default='once',
                        help='Execution mode')
    parser.add_argument('--data', help='Path to data file (CSV)')
    parser.add_argument('--target', help='Target column name')
    
    args = parser.parse_args()
    
    # Crear sistema
    system = AutoRetrainSystem(args.config)
    
    if args.mode == 'once':
        if args.data and args.target:
            import pandas as pd
            df = pd.read_csv(args.data)
            X = df.drop(columns=[args.target])
            y = df[args.target]
            
            results = system.run_cycle(X, y)
            print(json.dumps(results, indent=2))
        else:
            results = system.run_cycle()
            print(json.dumps(results, indent=2))
    else:
        if args.data and args.target:
            system.run_scheduled(args.data, args.target)
        else:
            print("Error: --data y --target requeridos para modo scheduled")


if __name__ == "__main__":
    main()