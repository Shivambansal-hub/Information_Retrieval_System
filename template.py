import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    ".env",
    "requirements.txt",
    "setup.py",
    "app.py",
    "research/trails.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:  # Only create directory if filedir is not empty
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Directory created: {filedir} for the file: {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as file:
            pass
        logging.info(f"File created: {filename}")
    else:
        logging.info(f"{filename} is already present in the directory")
