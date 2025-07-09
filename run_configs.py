"""
This is a simple shim script to run the specific configurations
with the proper import paths configured.
"""

import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now run the configs module
from training.run_specific_configs import main

if __name__ == "__main__":
    main()
