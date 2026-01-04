#!/bin/bash
# Quick setup and run script for Seed 2 forensic analysis

set -e

echo "=================================="
echo "Seed 2 Forensic Analysis - Setup"
echo "=================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "=================================="
echo "Running Forensic Analysis..."
echo "=================================="
echo ""

# Run the simplified analysis
python scripts/forensic_seed2_simple.py --detailed

echo ""
echo "=================================="
echo "Analysis Complete!"
echo "=================================="
echo "Results saved to: diagnostics/seed_forensics/"
echo ""
