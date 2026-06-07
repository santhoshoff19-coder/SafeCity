#!/bin/bash
echo "Downloading data from Google Drive..."
python3 -c "
import gdown
import zipfile
import os

if not os.path.exists('Data'):
    gdown.download('https://drive.google.com/uc?id=11hFq1TaDKPy-JkDl46pd2v2WmXHZNsL8', 'data.zip', quiet=False)
    print('Unzipping...')
    with zipfile.ZipFile('data.zip', 'r') as z:
        z.extractall('.')
    os.remove('data.zip')
    print('Data ready.')
else:
    print('Data already exists, skipping download.')
"
echo "Starting API..."
uvicorn main:app --host 0.0.0.0 --port $PORT
