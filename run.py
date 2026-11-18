"""
Product Sentiment Analysis — Quick Launcher
Starts the FastAPI backend and serves the fullstack application on http://localhost:8000
"""

import sys
import webbrowser
import uvicorn

import os
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DIST_INDEX = FRONTEND_DIR / "dist" / "index.html"

def ensure_frontend_built():
    """Ensure production frontend bundle exists before starting standalone server."""
    if DIST_INDEX.exists():
        return
    print("\n[*] Frontend production build not found. Building now...")
    npm_cmd = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
    try:
        subprocess.run([npm_cmd, "run", "build"], cwd=FRONTEND_DIR, check=True, shell=(sys.platform == "win32"))
        print("[+] Frontend build completed successfully.\n")
    except Exception as e:
        print(f"[!] Warning: Could not auto-build frontend ({e}).")
        print("    If you want the UI, run 'npm run build' inside the frontend directory.\n")

if __name__ == "__main__":
    ensure_frontend_built()

    print("=" * 60)
    print("  Product Sentiment Analysis — Full Stack AI Application")
    print("=" * 60)
    print("  Web App URL: http://localhost:8000")
    print("  API Docs:    http://localhost:8000/docs")
    print("  Press CTRL+C to stop the server.")
    print("=" * 60)
    
    # Optionally open browser automatically
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
