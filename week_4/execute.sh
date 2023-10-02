#!/bin/bash

# Create a virtual environment named 'brian_venv'
python3 -m venv brian_venv

# Activate the virtual environment
source brian_venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Execute the Python script
python3 nb_assignment_2.py freq_counts.csv --good_pdf 2 --bad_pdf 5 --test_pdf 4

# Deactivate the virtual environment
deactivate
