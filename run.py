"""
Product Sentiment Analysis — Quick Launcher
Starts the FastAPI backend and serves the fullstack application on http://localhost:8000
"""

import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  Product Sentiment Analysis — Full Stack AI Application")
    print("=" * 60)
    print("  URL: http://localhost:8000")
    print("  API Docs: http://localhost:8000/docs")
    print("  Press CTRL+C to stop the server.")
    print("=" * 60)
    
    # Optionally open browser automatically
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
