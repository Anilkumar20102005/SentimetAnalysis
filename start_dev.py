"""
Dual-Server Launcher: Starts FastAPI Backend and Vite React Frontend concurrently.
Handles graceful shutdown (Ctrl+C) and automatically launches the browser.
"""

import os
import sys
import time
import shutil
import signal
import threading
import subprocess
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

def find_npm():
    """Find npm executable, checking standard Windows installation paths if needed."""
    npm_path = shutil.which("npm.cmd") or shutil.which("npm")
    if npm_path:
        return npm_path
    
    # Common Windows paths
    candidates = [
        Path(r"C:\Program Files\nodejs\npm.cmd"),
        Path(r"C:\Program Files (x86)\nodejs\npm.cmd"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "nodejs" / "npm.cmd",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "npm"

def stream_logs(pipe, prefix, color_code):
    """Stream process stdout/stderr with clean colored tags."""
    reset = "\033[0m"
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            print(f"{color_code}{prefix}{reset} {line.rstrip()}", flush=True)
    except Exception:
        pass

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    os.system("")  # Enable ANSI escape sequences on Windows terminal
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print("=" * 65)
    print(f"{BOLD}  [*] Starting Full-Stack Sentiment Analysis Dev Servers{RESET}")
    print("=" * 65)
    print(f"  * {CYAN}FastAPI Backend:{RESET}   http://localhost:8000 (Docs: /docs)")
    print(f"  * {GREEN}React Frontend:{RESET}    http://localhost:5173")
    print("  * Press CTRL+C at any time to terminate both servers.")
    print("=" * 65 + "\n")

    # Update PATH to include nodejs if installed in standard location
    node_dir = Path(r"C:\Program Files\nodejs")
    if node_dir.exists():
        os.environ["PATH"] = str(node_dir) + os.path.pathsep + os.environ.get("PATH", "")

    npm_bin = find_npm()

    # 1. Start FastAPI Backend
    backend_cmd = [
        sys.executable,
        "-m", "uvicorn",
        "backend.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # 2. Start React Vite Frontend
    frontend_cmd = [npm_bin, "run", "dev"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream logs asynchronously
    t1 = threading.Thread(target=stream_logs, args=(backend_proc.stdout, "[BACKEND ]", CYAN), daemon=True)
    t2 = threading.Thread(target=stream_logs, args=(frontend_proc.stdout, "[FRONTEND]", GREEN), daemon=True)
    t1.start()
    t2.start()

    # Open browser once Vite is ready
    def open_browser():
        time.sleep(2.5)
        webbrowser.open("http://localhost:5173")

    threading.Thread(target=open_browser, daemon=True).start()

    # Monitor processes & wait for Ctrl+C
    try:
        while True:
            b_poll = backend_proc.poll()
            f_poll = frontend_proc.poll()
            if b_poll is not None:
                print(f"{YELLOW}\n[BACKEND ] Process exited with code {b_poll}{RESET}")
                break
            if f_poll is not None:
                print(f"{YELLOW}\n[FRONTEND] Process exited with code {f_poll}{RESET}")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Shutting down servers...{RESET}")
    finally:
        for proc, name in [(backend_proc, "Backend"), (frontend_proc, "Frontend")]:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        print(f"{GREEN}Both servers stopped cleanly.{RESET}")

if __name__ == "__main__":
    main()
