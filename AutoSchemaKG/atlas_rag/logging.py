import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
import os
import datetime
from atlas_rag.evaluation.benchmark import BenchMarkConfig

def setup_logger(config:BenchMarkConfig, logger_name = "MyLogger", log_path = None) -> Logger:
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    if log_path:
        log_file_path = log_path
    else:
        log_file_path = f'./log/{config.dataset_name}_event{config.include_events}_concept{config.include_concept}_{date_time}.log'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    max_bytes = 50 * 1024 * 1024 
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=5)
    logger.addHandler(handler)
    
    return logger