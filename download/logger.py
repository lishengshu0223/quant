import os
import time
from loguru import logger as loguru_logger
from .config import DEFAULT_DATA_ROOT


def setup_logger(name="download_logger"):
    log_dir = os.path.join(DEFAULT_DATA_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"download_{time.strftime('%Y%m%d')}.log")
    
    loguru_logger.remove()
    
    loguru_logger.add(
        log_file,
        level="DEBUG",
        format="{time} - {name} - {level} - {message}",
        encoding="utf-8",
        enqueue=True,
    )
    
    loguru_logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="{time} - {name} - {level} - {message}",
        enqueue=True,
    )
    
    return loguru_logger


logger = setup_logger()