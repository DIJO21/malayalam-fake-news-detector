import os
import sys
import subprocess
import json

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import('PyPDF2')
install_and_import('pandas')

import PyPDF2
import pandas as pd

pdf_path = "project_report (8).pdf"
csv_path = "data/raw/malayalam_dataset_fast.csv"

# Extract PDF
try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for i in range(len(reader.pages)):
            text += reader.pages[i].extract_text() + "\n"
        with open("report_text.txt", "w", encoding="utf-8") as out:
            out.write(text)
    print("PDF extracted successfully.")
except Exception as e:
    print(f"Error extracting PDF: {e}")

# Extract CSV info
try:
    df = pd.read_csv(csv_path)
    info = f"Columns: {list(df.columns)}\n\nFirst 5 rows:\n{df.head().to_string()}\n\nLabel counts:\n"
    
    # Try to find label column
    label_col = None
    if 'label' in df.columns: label_col = 'label'
    elif 'Label' in df.columns: label_col = 'Label'
    elif 'class' in df.columns: label_col = 'class'
    elif 'target' in df.columns: label_col = 'target'
    elif 'Fake/Real' in df.columns: label_col = 'Fake/Real'
    
    if label_col:
        info += df[label_col].value_counts().to_string()
    else:
        info += "Could not automatically identify label column."

    with open("dataset_info.txt", "w", encoding="utf-8") as out:
        out.write(info)
    print("CSV info extracted successfully.")
except Exception as e:
    print(f"Error extracting CSV: {e}")
