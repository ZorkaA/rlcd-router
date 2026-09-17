#!/usr/bin/env python
"""Entry-point script for the MoE Router Calibration Pipeline.

Usage:
    python scripts/run_calibration.py --synthetic --verbose
    python scripts/run_calibration.py --device cpu --output results/report.json
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import main

if __name__ == "__main__":
    report = main()
    print(f"\nPipeline finished. Overall ECE: {report.get('phases', {}).get('evaluation', {}).get('overall_ece', 'N/A')}")
