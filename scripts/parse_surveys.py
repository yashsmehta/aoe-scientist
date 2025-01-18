import os
import time
import re
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

survey_dir = Path("data/surveys/neuroai")

converter = PdfConverter(artifact_dict=create_model_dict())

for pdf_path in survey_dir.glob("*.pdf"):
    txt_path = pdf_path.with_suffix('.txt')
    
    # Skip if output file already exists
    if txt_path.exists():
        print(f"Skipping {pdf_path.name} - already processed")
        continue
    start_time = time.time()
        
    print(f"Processing {pdf_path.name}...")
    rendered = converter(str(pdf_path))
    text, _, images = text_from_rendered(rendered)
    
    txt_path.write_text(text)
    end_time = time.time()
    print(f"{pdf_path.name} extraction completed in {end_time - start_time:.2f} seconds\n")

