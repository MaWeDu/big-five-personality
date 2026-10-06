import os
import subprocess
import sys
from pathlib import Path

def is_in_venv():
    return (
        hasattr(sys, "real_prefix") or
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix) or
        os.environ.get("VIRTUAL_ENV") is not None
    )

def main():
    # Projekt-Root ist eine Ebene über dem src-Ordner
    base_dir = Path(__file__).resolve().parent.parent
    src_dir = Path(__file__).resolve().parent
    venv_dir = base_dir / ".venv"

    if os.name == "nt":  # Windows
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:  # macOS / Linux
        venv_python = venv_dir / "bin" / "python"

    venv_exists = venv_python.exists()

    if not is_in_venv():
        if not venv_exists:
            print("----------------------------------------------------------------")
            print(" Hinweis: Es läuft aktuell keine virtuelle Umgebung (venv).")
            print("----------------------------------------------------------------")
            choice = input("Virtuelle Umgebung (.venv) im Projektordner erstellen? [J/n]: ").strip().lower()
            
            if choice in ["", "j", "yes", "ja"]:
                print(f"\n>>> Erstelle virtuelle Umgebung in '{venv_dir}'...")
                subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

                print(">>> Installiere Abhängigkeiten aus 'requirements.txt'...")
                subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
                subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(base_dir / "requirements.txt")], check=True)
            else:
                print(">>> Verwende das System-Python.")

        if venv_python.exists():
            print("----------------------------------------------------------------")
            print(" Starte den Workflow in der virtuellen Umgebung (.venv)...")
            print("----------------------------------------------------------------")
            result = subprocess.run([str(venv_python)] + sys.argv)
            sys.exit(result.returncode)

    force_run = "--force" in sys.argv

    # Schritt 1: EDA & Datenbereinigung
    clean_data_path = base_dir / "DATA" / "data_clean.csv"
    if not clean_data_path.exists() or force_run:
        print("\n=== SCHRITT 1: Starte EDA & Datenbereinigung ===")
        subprocess.run([sys.executable, str(src_dir / "01_eda.py")], check=True)
    else:
        print("\n=== SCHRITT 1: Übersprungen (Bereinigte Daten 'data_clean.csv' existieren bereits) ===")

    # Schritt 2: Modell-Training & Export
    model_path = base_dir / "best_model.joblib"
    if not model_path.exists() or force_run:
        print("\n=== SCHRITT 2: Starte Modell-Training, Tuning & Export ===")
        subprocess.run([sys.executable, str(src_dir / "02_train.py")], check=True)
    else:
        print("\n=== SCHRITT 2: Übersprungen (Trainiertes Modell 'best_model.joblib' existiert bereits) ===")

    # Schritt 3: Streamlit App starten
    print("\n=== SCHRITT 3: Starte Streamlit App ===")
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(base_dir / "app.py")], 
        env=env, 
        check=True
    )

if __name__ == "__main__":
    main()