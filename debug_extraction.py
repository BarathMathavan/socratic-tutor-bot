# debug_extraction.py
from ingestion import extract_text_from_pdf

# --- IMPORTANT: Change this to the exact name of your PDF file ---
PDF_FILENAME = "Motor_Vehicles_Act_1988_with_Amendments_2019.pdf"
# -------------------------------------------------------------

filepath = f"uploads/{PDF_FILENAME}"

print(f"--- EXTRACTING TEXT FROM {filepath} ---")
raw_text = extract_text_from_pdf(filepath)
print("--- EXTRACTION COMPLETE ---")

# Save the extracted text to a file so you can inspect it
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)

print("\nSuccessfully extracted text and saved it to 'extracted_text.txt'.")
print("Open that file and search it to see how the computer sees your PDF.")