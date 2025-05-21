# torchklip/config.py
import os
from pathlib import Path

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parents[2]

# Default configurations
DEFAULT_LOG_DIR = PROJECT_ROOT / 'logs'

# Environment variable overrides default if provided
LOG_DIR = Path(os.environ.get('TORCHKLIP_LOG_DIR', DEFAULT_LOG_DIR))
