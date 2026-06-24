import logging
import sys

def get_bashar_logger(name="BASHAR"):
    """
    Standardized telemetry logger for the BASHAR library.
    Ensures all terminal output is cleanly formatted and easy to debug.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if the logger is imported multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Format: [BASHAR] [INFO]: Message
        formatter = logging.Formatter('[%(name)s] [%(levelname)s]: %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger