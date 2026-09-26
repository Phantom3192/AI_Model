"""
launcher.py - Installs requirements first (using a big temp folder instead of
the small /tmp), then starts server.py.

Put this file in the same folder as server.py and requirement.txt, and set
launcher.py as the file to run (instead of server.py).
"""
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
REQ = BASE / "requirement.txt"
MAIN = BASE / "server.py"
MARKER = BASE / ".deps_installed_v2"
BUILD_DIR = BASE / ".build_tmp"

for sub in ("tmp", "pip_cache"):
    (BUILD_DIR / sub).mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["TMPDIR"] = str(BUILD_DIR / "tmp")
env["TEMP"] = str(BUILD_DIR / "tmp")
env["TMP"] = str(BUILD_DIR / "tmp")
env["PIP_CACHE_DIR"] = str(BUILD_DIR / "pip_cache")


def pip_install(req_file: Path) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    print(f"[launcher] Running: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, env=env, cwd=BASE) == 0


def install_dependencies() -> None:
    if MARKER.exists():
        print("[launcher] Dependencies already installed, skipping.", flush=True)
        return

    print("[launcher] Installing dependencies (first run can take a while)...", flush=True)
    if pip_install(REQ):
        MARKER.write_text("ok")
        return

    print("[launcher] Install failed. Check the errors above.", flush=True)
    sys.exit(1)


def load_dotenv_and_report() -> None:
    dotenv_path = BASE / ".env"
    if dotenv_path.exists():
        loaded = 0
        for line in dotenv_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in env:
                env[key] = value
                loaded += 1
        print(f"[launcher] .env found: {dotenv_path} ({loaded} settings loaded)", flush=True)
    else:
        print(f"[launcher] .env NOT found at {dotenv_path} (using panel variables only)", flush=True)

    for key in ("API_KEY", "BANK_DIR", "HF_TOKEN"):
        print(f"[launcher]   {key}: {'SET' if env.get(key) else 'missing'}", flush=True)


def main() -> None:
    install_dependencies()
    load_dotenv_and_report()
    print("[launcher] Starting server.py ...", flush=True)
    sys.exit(subprocess.call([sys.executable, str(MAIN)], env=env, cwd=BASE))


if __name__ == "__main__":
    main()