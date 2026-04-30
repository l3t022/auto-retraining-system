"""
Trainer Module
Entrena modelos con búsqueda de hiperparámetros usando Optuna.
"""

import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, train_test_split


class ModelTrainer:
    """Entrena modelos con optimización de hiperparámetros via Optuna."""
    
    def __init__(
        self,
        task_type: str = "regression",
        model_type: str = "xgboost",
        n_trials: int = 20,
        timeout: int = 300,
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.task_type = task_type
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.study = None
    
    def _get_objective(
        self, X: np.ndarray, y: np.ndarray
    ) -> Callable:
        """Crea la función objetivo para Optuna."""
        
        def objective(trial: optuna.Trial) -> float:
            # Definir search space según el modelo
            if self.model_type == "xgboost":
                params = self._xgboost_suggest(trial)
                model = self._create_xgboost(params)
            elif self.model_type == "lightgbm":
                params = self._lightgbm_suggest(trial)
                model = self._create_lightgbm(params)
            elif self.model_type == "random_forest":
                params = self._random_forest_suggest(trial)
                model = self._create_random_forest(params)
            else:
                raise ValueError(f"Modelo no soportado: {self.model_type}")
            
            # Cross-validation
            if self.task_type == "regression":
                scoring = "neg_mean_squared_error"
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=scoring)
                score = -scores.mean()  # Negar porque es neg_mse
            else:
                scoring = "accuracy"
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=scoring)
                score = scores.mean()
            
            return score
        
        return objective
    
    def _xgboost_suggest(self, trial: optuna.Trial) -> Dict:
        """Sugiere hiperparámetros para XGBoost."""
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": self.random_state,
        }
        
        if self.task_type == "classification":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "reg:squarederror"
        
        return params
    
    def _lightgbm_suggest(self, trial: optuna.Trial) -> Dict:
        """Sugiere hiperparámetros para LightGBM."""
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": self.random_state,
            "verbose": -1,
        }
        
        if self.task_type == "classification":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
        else:
            params["objective"] = "regression"
            params["metric"] = "rmse"
        
        return params
    
    def _random_forest_suggest(self, trial: optuna.Trial) -> Dict:
        """Sugiere hiperparámetros para Random Forest."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        return params
    
    def _create_xgboost(self, params: Dict) -> object:
        """Crea modelo XGBoost."""
        from xgboost import XGBClassifier, XGBRegressor
        
        params = {k: v for k, v in params.items() if k not in ["objective", "eval_metric"]}
        
        if self.task_type == "classification":
            return XGBClassifier(**params)
        else:
            return XGBRegressor(**params)
    
    def _create_lightgbm(self, params: Dict) -> object:
        """Crea modelo LightGBM."""
        from lightgbm import LGBMClassifier, LGBMRegressor
        
        params = {k: v for k, v in params.items() if k not in ["objective", "metric", "verbose"]}
        
        if self.task_type == "classification":
            return LGBMClassifier(**params)
        else:
            return LGBMRegressor(**params)
    
    def _create_random_forest(self, params: Dict) -> object:
        """Crea modelo Random Forest."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if self.task_type == "classification":
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[object, Dict, float]:
        """Entrena el modelo con optimización de hiperparámetros."""
        
        # Crear study de Optuna
        direction = "minimize" if self.task_type == "regression" else "maximize"
        sampler = TPESampler(seed=self.random_state)
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=f"{self.model_type}_{self.task_type}",
        )
        
        # Crear función objetivo
        objective = self._get_objective(X, y)
        
        # Ejecutar optimización
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        
        # Obtener mejores hiperparámetros
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        # Entrenar modelo final con mejores hiperparámetros
        if self.model_type == "xgboost":
            full_params = self.best_params.copy()
            if self.task_type == "classification":
                full_params["objective"] = "binary:logistic"
            else:
                full_params["objective"] = "reg:squarederror"
            full_params["random_state"] = self.random_state
            self.best_model = self._create_xgboost(full_params)
            
        elif self.model_type == "lightgbm":
            full_params = self.best_params.copy()
            if self.task_type == "classification":
                full_params["objective"] = "binary"
            else:
                full_params["objective"] = "regression"
            full_params["random_state"] = self.random_state
            self.best_model = self._create_lightgbm(full_params)
            
        elif self.model_type == "random_forest":
            self.best_model = self._create_random_forest(self.best_params)
        
        # Entrenar en todos los datos
        self.best_model.fit(X, y)
        
        return self.best_model, self.best_params, self.best_score
    
    def train_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict,
    ) -> object:
        """Entrena un modelo con params específicos (sin búsqueda)."""
        
        if self.model_type == "xgboost":
            full_params = params.copy()
            if self.task_type == "classification":
                full_params["objective"] = "binary:logistic"
            else:
                full_params["objective"] = "reg:squarederror"
            full_params["random_state"] = self.random_state
            model = self._create_xgboost(full_params)
        elif self.model_type == "lightgbm":
            full_params = params.copy()
            if self.task_type == "classification":
                full_params["objective"] = "binary"
            else:
                full_params["objective"] = "regression"
            full_params["random_state"] = self.random_state
            model = self._create_lightgbm(full_params)
        else:
            model = self._create_random_forest(params)
        
        model.fit(X, y)
        return model
    
    def get_study_results(self) -> Dict:
        """Retorna resultados del estudio Optuna."""
        if self.study is None:
            return {}
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.study.trials),
            "best_trial": self.study.best_trial.number,
            "trials_data": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in self.study.trials
            ],
        }
    
    def save_model(self, path: str):
        """Guarda el modelo entrenado."""
        if self.best_model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.best_model, path)
    
    def load_model(self, path: str) -> object:
        """Carga un modelo guardado."""
        self.best_model = joblib.load(path)
        return self.best_model


def train_with_optuna(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "xgboost",
    task_type: str = "regression",
    n_trials: int = 20,
    timeout: int = 300,
    cv_folds: int = 3,
    random_state: int = 42,
) -> Tuple[object, Dict, float, optuna.Study]:
    """Función de conveniencia para entrenar con Optuna."""
    
    trainer = ModelTrainer(
        model_type=model_type,
        task_type=task_type,
        n_trials=n_trials,
        timeout=timeout,
        cv_folds=cv_folds,
        random_state=random_state,
    )
    
    model, params, score = trainer.train(X, y)
    
    return model, params, score, trainer.study