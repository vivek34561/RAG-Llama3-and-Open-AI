import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO , format='[%(asctime)s]: %(message)s')

lists_of_files = [
    "uploaded_documents/",
    ".env",
    "app.py",
    "requirements.txt",
]

for filepath in lists_of_files:
    filepath = Path(filepath)
    filedir , filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir , exist_ok=True)
        logging.info(f"Creating directory {filedir} for the file {filename}")   
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath , 'w') as f:
            pass
        logging.info(f"Creating empty file {filename} in the directory {filedir}")
    else:
        logging.info(f"File {filename} already exists in the directory {filedir}")
        