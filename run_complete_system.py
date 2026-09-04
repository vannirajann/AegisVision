"""
AegisVision - Complete Integrated System Runner

Starts all services:
- Backend API (FastAPI)
- Surveillance Pipeline (Detection + Analytics)
- Detection API (Optional)
"""

import subprocess
import sys
import os
import time
import threading

def run_backend():
    """Run backend API server"""
    print("[BACKEND] Starting FastAPI server on port 8000...")
    original_cwd = os.getcwd()
    try:
        os.chdir(os.path.join(original_cwd, "backend"))
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", "--reload", "--port", "8000"
        ])
    finally:
        os.chdir(original_cwd)

def run_pipeline():
    """Run main surveillance pipeline"""
    print("[PIPELINE] Waiting for backend to start...")
    time.sleep(3)  # Wait for backend to start
    print("[PIPELINE] Starting surveillance pipeline...")
    original_cwd = os.getcwd()
    try:
        os.chdir(os.path.join(original_cwd, "pipeline"))
        subprocess.run([sys.executable, "surveillance_pipeline.py"])
    finally:
        os.chdir(original_cwd)

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   🛡 AegisVision - Complete Integrated System            ║
║                                                          ║
║   Starting all services...                               ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Start backend in background thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Start pipeline in main thread (to see output)
    run_pipeline()

if __name__ == "__main__":
    main()
