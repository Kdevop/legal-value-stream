import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def setup_audit_logger(log_file='ai_audit.log', retention_months=12):
    """
    Sets up an audit logger with monthly rotation and 12-month retention.
    Logs are rotated on the first day of each month, keeping up to 12 backup files.
    """
    logger = logging.getLogger('ai_audit')
    logger.setLevel(logging.INFO)

    # Handler rotates monthly ('M'), keeps 12 backups (12 months)
    handler = TimedRotatingFileHandler(
        log_file,
        when='M',  # Rotate monthly
        interval=1,
        backupCount=retention_months  # Retain 12 months
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def log_ai_interaction(logger, user_input, ai_output, model_name, metadata=None):
    """
    Logs an AI interaction with timestamp, model, input, output, and optional metadata.
    """
    entry = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'input': user_input,
        'output': ai_output,
        'metadata': metadata or {}
    }
    logger.info(f"AI Interaction: {entry}")