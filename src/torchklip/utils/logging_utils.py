# torchklip/utils/logging_utils.py
import os
import logging
import datetime
from pathlib import Path
from ..config import LOG_DIR

# Define format constants
BASIC_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
JUPYTER_FORMAT = '%(message)s'


def setup_logger(log_dir=None, log_level=logging.INFO, console_level=logging.INFO,
                 jupyter_mode=None, show_logger_name=False):
    """
    Set up a logger that writes to both a file with timestamp and the console.

    Args:
        log_dir (str, optional): Directory to store log files. If None, uses the configured LOG_DIR.
        log_level (int, optional): Logging level for the file handler. Default is INFO.
        console_level (int, optional): Logging level for the console handler. Default is INFO.
        jupyter_mode (bool, optional): Force Jupyter notebook formatting. If None, auto-detects.
        show_logger_name (bool, optional): Whether to show logger name in console output. Default is False.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('torchklip')
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything

    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Auto-detect if running in Jupyter if not specified
    if jupyter_mode is None:
        try:
            from IPython import get_ipython
            jupyter_mode = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
        except (ImportError, AttributeError):
            jupyter_mode = False

    # Create formatters
    # Always use full format for file logs
    file_formatter = logging.Formatter(SIMPLE_FORMAT)

    # Choose console formatter based on settings
    if jupyter_mode:
        # Jupyter mode - just show the message
        console_formatter = logging.Formatter(JUPYTER_FORMAT)
    elif show_logger_name:
        # Show full format including logger name
        console_formatter = logging.Formatter(BASIC_FORMAT)
    else:
        # Show timestamp and level but no logger name
        console_formatter = logging.Formatter(SIMPLE_FORMAT)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Use configured log_dir if none provided
    if log_dir is None:
        log_dir = LOG_DIR
    else:
        log_dir = Path(log_dir)

    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique log filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'torchklip_{timestamp}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def get_logger(name=None):
    """
    Get a logger that inherits from the main torchklip logger.

    Args:
        name (str, optional): Name for the logger, typically __name__. 
                             If None, returns the main torchklip logger.

    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        return logging.getLogger('torchklip')
    return logging.getLogger(f'torchklip.{name}')


__all__ = ["setup_logger", "get_logger"]
