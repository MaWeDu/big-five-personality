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
    # run.py, 01_eda.py and 02_train.py are in src.
    src_dir = Path(__file__).resolve().parent

    # app.py and requirements.txt are in the project root.
    project_dir = src_dir.parent

    # Keep the virtual environment, generated data and models in src.
    venv_dir = src_dir / ".venv"

    if os.name == "nt":  # Windows
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:  # macOS / Linux
        venv_python = venv_dir / "bin" / "python"

    # Create and use a virtual environment when needed.
    if not is_in_venv():
        if not venv_python.exists():
            print("----------------------------------------------------------------")
            print(" Note: No virtual environment is currently active.")
            print("----------------------------------------------------------------")

            choice = input(
                "Create a virtual environment in the src folder? [Y/n]: "
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
                        str(project_dir / "requirements.txt"),
                    ],
                    check=True,
                )
            else:
                print(">>> Using the system Python installation.")

        if venv_python.exists():
            print("----------------------------------------------------------------")
            print(" Starting the workflow in the virtual environment (.venv)...")
            print("----------------------------------------------------------------")

            try:
                result = subprocess.run(
                    [str(venv_python)] + sys.argv,
                    cwd=src_dir,
                )
            except KeyboardInterrupt:
                print("\nWorkflow stopped.")
                return

            sys.exit(result.returncode)

    force_run = "--force" in sys.argv

    # Generated files are stored inside src.
    clean_data_path = src_dir / "data" / "data_clean.csv"
    model_path = src_dir / "models" / "personality_pipeline.joblib"

    # Step 1: EDA and data cleaning
    if not clean_data_path.exists() or force_run:
        print("\n=== STEP 1: Starting EDA and data cleaning ===")
        subprocess.run(
            [sys.executable, str(src_dir / "01_eda.py")],
            cwd=src_dir,
            check=True,
        )
    else:
        print("\n=== STEP 1: Skipped (cleaned data already exists) ===")

    # Step 2: Model training and export
    if not model_path.exists() or force_run:
        print("\n=== STEP 2: Starting model training, tuning, and export ===")
        subprocess.run(
            [sys.executable, str(src_dir / "02_train.py")],
            cwd=src_dir,
            check=True,
        )
    else:
        print("\n=== STEP 2: Skipped (trained model already exists) ===")

    # Step 3: Start the Streamlit app
    print("\n=== STEP 3: Starting the Streamlit app ===")

    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(project_dir / "app.py"),
            ],
            cwd=src_dir,
            env=env,
            check=True,
        )
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")


if __name__ == "__main__":
    main()