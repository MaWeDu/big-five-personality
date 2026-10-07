# Write requirements.txt with the exact versions installed in THIS environment
from importlib.metadata import version, PackageNotFoundError

# Every library the project imports (notebooks + app.py), plus Jupyter to run the notebooks
packages = [
    "pandas", "numpy", "matplotlib", "seaborn",     # data + plots
    "scikit-learn", "hyperopt", "joblib", "tqdm",   # modeling
    "streamlit",                                    # the app
    "gdown", "ipywidgets",                          # data download in eda.ipynb
    "notebook", "ipykernel",                        # to run the notebooks
]

lines = []
for pkg in packages:
    try:
        lines.append(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        print(f"⚠️ {pkg} is not installed here - run: pip install {pkg}")

with open("requirements.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("requirements.txt written:\n")
print("\n".join(lines))