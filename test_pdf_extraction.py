#!/usr/bin/env python3
"""
Test script to verify PDF extraction works with FontBBox error handling.
"""

import os
import sys
from main import extract_text_from_pdf

def test_pdf_extraction():
    """Test PDF extraction on files in the data directory."""
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please create it and add some PDF files.")
        return
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to test:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file}")
    
    print("\nTesting PDF extraction...")
    
    for pdf_file in pdf_files:
        print(f"\n{'='*50}")
        print(f"Testing: {pdf_file}")
        print('='*50)
        
        pdf_path = os.path.join(data_dir, pdf_file)
        
        try:
            pages = extract_text_from_pdf(pdf_path)
            print(f"Successfully extracted {len(pages)} pages")
            
            # Show first few characters of each page
            for page_num, text in pages[:3]:  # Show first 3 pages
                preview = text[:100].replace('\n', ' ').strip()
                print(f"  Page {page_num}: {preview}...")
            
            if len(pages) > 3:
                print(f"  ... and {len(pages) - 3} more pages")
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    test_pdf_extraction()
