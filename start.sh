#!/bin/bash

# Navigate to the project folder
cd "$(dirname "$0")"

# Activate the correct conda environment
eval "$(conda shell.bash hook)"
conda activate chatbot

# Start the FastAPI app
python chatbot_api.py