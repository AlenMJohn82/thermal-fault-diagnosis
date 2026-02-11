#!/bin/bash

# Activation script for thermal conda environment

echo "==============================================="
echo "  Thermal Fault Diagnosis - Environment Setup"
echo "==============================================="
echo ""
echo "Activating conda environment: thermal"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thermal

echo ""
echo "✓ Environment activated!"
echo ""
echo "Available commands:"
echo "  python train.py              - Train the model"
echo "  python app.py                - Run web UI"
echo ""
echo "To deactivate, run: conda deactivate"
echo "==============================================="
