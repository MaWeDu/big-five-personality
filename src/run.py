import os
import subprocess
import sys
from pathlib import Path


def is_in_venv():
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def main():
    # The project root is one level above the src folder.
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
            print(" Note: No virtual environment (venv) is currently active.")
            print("----------------------------------------------------------------")
            choice = input(
                "Create a virtual environment (.venv) in the project folder? [Y/n]: "
            ).strip().lower()

            if choice in ["", "y", "yes"]:
                print(f"\n>>> Creating virtual environment in '{venv_dir}'...")
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    check=True,
                )

                print(">>> Installing dependencies from 'requirements.txt'...")
                subprocess.run(
                    [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                    check=True,
                )
                subprocess.run(
                    [
                        str(venv_python),
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(base_dir / "requirements.txt"),
                    ],
                    check=True,
                )
            else:
                print(">>> Using the system Python installation.")

        if venv_python.exists():
            print("----------------------------------------------------------------")
            print(" Starting the workflow in the virtual environment (.venv)...")
            print("----------------------------------------------------------------")
            result = subprocess.run([str(venv_python)] + sys.argv)
            sys.exit(result.returncode)

    force_run = "--force" in sys.argv

    # Step 1: EDA and data cleaning
    clean_data_path = base_dir / "DATA" / "data_clean.csv"
    if not clean_data_path.exists() or force_run:
        print("\n=== STEP 1: Starting EDA and data cleaning ===")
        subprocess.run(
            [sys.executable, str(src_dir / "01_eda.py")],
            check=True,
        )
    else:
        print(
            "\n=== STEP 1: Skipped "
            "(cleaned data file 'data_clean.csv' already exists) ==="
        )

    # Step 2: Model training and export
    model_path = base_dir / "best_model.joblib"
    if not model_path.exists() or force_run:
        print("\n=== STEP 2: Starting model training, tuning, and export ===")
        subprocess.run(
            [sys.executable, str(src_dir / "02_train.py")],
            check=True,
        )
    else:
        print(
            "\n=== STEP 2: Skipped "
            "(trained model file 'best_model.joblib' already exists) ==="
        )

    # Step 3: Start the Streamlit app
    print("\n=== STEP 3: Starting the Streamlit app ===")
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(base_dir / "app.py")],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
