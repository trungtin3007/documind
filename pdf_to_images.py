import fitz  # PyMuPDF
import os

PDF_PATH = "sample.pdf"   # your PDF
OUT_DIR = "pages"         # where images go

os.makedirs(OUT_DIR, exist_ok=True)
doc = fitz.open(PDF_PATH)

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)                     # render at 150 DPI
    out_path = os.path.join(OUT_DIR, f"page_{i+1:03d}.png")
    pix.save(out_path)
    print(f"saved {out_path}")

print(f"\nDone: {len(doc)} pages -> {OUT_DIR}/")
doc.close()