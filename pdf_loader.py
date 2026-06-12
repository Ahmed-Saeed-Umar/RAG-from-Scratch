import pdfplumber

def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    full_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # some pages are images — skip them
                full_text.append(text)
    return "\n\n".join(full_text)