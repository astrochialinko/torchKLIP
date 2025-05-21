
import os
import sys
import pytest

# Add the src directory to the path so that imports work correctly
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))
