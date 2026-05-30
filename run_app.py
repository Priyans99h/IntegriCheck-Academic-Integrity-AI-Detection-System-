"""
IntegriCheck — Quick Start
Run this file to start the Flask web application.
Usage: python run_app.py
Then open: http://localhost:5000
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flask_app.app import app

if __name__ == '__main__':
    print('Starting IntegriCheck...')
    print('Open browser: http://localhost:5000')
    print('Dashboard:    http://localhost:5000/dashboard')
    print('Press Ctrl+C to stop.')
    app.run(host='0.0.0.0', port=5000, debug=False)
