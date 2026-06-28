# src/utils/logger.py
import logging
import sys

def setup_logger():
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if reloaded
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Professional formatting with timestamps
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()