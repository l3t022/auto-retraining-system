"""
Monitor Module
Monitorea nuevos datos y detecta cuando es necesario evaluar el modelo.
"""

import os
import time
from typing import Dict, List, Optional, Callable
from pathlib import Path

import schedule


class DataMonitor:
    """Monitorea directorios/archivos para detectar nuevos datos."""
    
    def __init__(self, watch_paths: List[str], check_interval_seconds: int = 60):
        self.watch_paths = watch_paths
        self.check_interval = check_interval_seconds
        self.file_states = {}
        self.callbacks = []
        
        # Inicializar estados
        for path in watch_paths:
            self._update_file_state(path)
    
    def _update_file_state(self, path: str):
        """Actualiza el estado de un archivo/directorio."""
        if os.path.isfile(path):
            stat = os.stat(path)
            self.file_states[path] = {
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'type': 'file',
            }
        elif os.path.isdir(path):
            files = []
            total_size = 0
            max_mtime = 0
            
            for root, dirs, filenames in os.walk(path):
                for f in filenames:
                    fpath = os.path.join(root, f)
                    try:
                        stat = os.stat(fpath)
                        total_size += stat.st_size
                        max_mtime = max(max_mtime, stat.st_mtime)
                        files.append(fpath)
                    except:
                        pass
            
            self.file_states[path] = {
                'total_size': total_size,
                'max_mtime': max_mtime,
                'type': 'directory',
                'files': files,
            }
    
    def has_changed(self, path: str) -> bool:
        """Detecta si un archivo/directorio cambió."""
        if path not in self.file_states:
            return True
        
        old_state = self.file_states[path]
        
        if old_state['type'] == 'file':
            try:
                stat = os.stat(path)
                return (
                    stat.st_size != old_state['size'] or
                    stat.st_mtime != old_state['mtime']
                )
            except:
                return True
        else:  # directory
            try:
                current_state = self._get_directory_state(path)
                return (
                    current_state['total_size'] != old_state.get('total_size', 0) or
                    current_state['max_mtime'] != old_state.get('max_mtime', 0)
                )
            except:
                return True
    
    def _get_directory_state(self, path: str) -> Dict:
        """Obtiene el estado actual de un directorio."""
        files = []
        total_size = 0
        max_mtime = 0
        
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                fpath = os.path.join(root, f)
                try:
                    stat = os.stat(fpath)
                    total_size += stat.st_size
                    max_mtime = max(max_mtime, stat.st_mtime)
                    files.append(fpath)
                except:
                    pass
        
        return {
            'total_size': total_size,
            'max_mtime': max_mtime,
            'files': files,
        }
    
    def check_all(self) -> Dict[str, bool]:
        """Verifica todos los paths."""
        results = {}
        
        for path in self.watch_paths:
            results[path] = self.has_changed(path)
            if results[path]:
                self._update_file_state(path)
        
        return results
    
    def register_callback(self, callback: Callable):
        """Registra un callback para cuando se detectan cambios."""
        self.callbacks.append(callback)
    
    def run_once(self) -> bool:
        """Ejecuta una verificación."""
        changes = self.check_all()
        
        has_any_change = any(changes.values())
        
        if has_any_change:
            for callback in self.callbacks:
                callback(changes)
        
        return has_any_change
    
    def run_continuously(self, stop_event=None):
        """Ejecuta monitoreo continuo."""
        import threading
        
        def run():
            while True:
                self.run_once()
                time.sleep(self.check_interval)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
        return thread


class ScheduledMonitor:
    """Monitoreo basado en schedule (cron)."""
    
    def __init__(self):
        self.jobs = []
    
    def schedule_daily(self, time_str: str, callback: Callable):
        """Agenda tarea diaria."""
        # time_str format: "HH:MM"
        job = schedule.every().day.at(time_str)
        job.do(callback)
        self.jobs.append(job)
    
    def schedule_hours(self, hours: int, callback: Callable):
        """Agenda tarea cada N horas."""
        job = schedule.every(hours).hours
        job.do(callback)
        self.jobs.append(job)
    
    def schedule_minutes(self, minutes: int, callback: Callable):
        """Agenda tarea cada N minutos."""
        job = schedule.every(minutes).minutes
        job.do(callback)
        self.jobs.append(job)
    
    def run_pending(self):
        """Ejecuta tareas pendientes."""
        schedule.run_pending()
    
    def run_continuously(self, interval_seconds: int = 60):
        """Ejecuta el scheduler continuamente."""
        import threading
        
        def run():
            while True:
                self.run_pending()
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
        return thread


# Funciones de conveniencia para Google Colab

def create_drive_monitor(file_id: str, check_interval: int = 3600):
    """Crea un monitor para Google Drive."""
    # En Colab, estocheck_interval la mount de Drive
    return DataMonitor(
        watch_paths=[f"/content/drive/MyDrive/{file_id}"],
        check_interval_seconds=check_interval,
    )


def detect_csv_changes(csv_path: str, hash_file: str = None) -> bool:
    """Simple detección de cambios en CSV."""
    import hashlib
    
    if not os.path.exists(csv_path):
        return False
    
    # Calcular hash actual
    with open(csv_path, 'rb') as f:
        current_hash = hashlib.md5(f.read()).hexdigest()
    
    # Cargar hash previo
    hash_file = hash_file or csv_path + ".hash"
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            old_hash = f.read()
        
        if current_hash != old_hash:
            # Guardar nuevo hash
            with open(hash_file, 'w') as f:
                f.write(current_hash)
            return True
    
    # Primer uso - guardar hash
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    
    return False