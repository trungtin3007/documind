import os, json, glob
import numpy as np
import voyageai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()               # reads VOYAGE_API_KEY from .env
vo = voyageai.Client()      # picks up the key automatically

PAGES_DIR = "pages"
OUT_DIR = "index"
MODEL = "voyage-multimodal-3"

os.makedirs(OUT_DIR, exist_ok=True)
page_files = sorted(glob.glob(os.path.join(PAGES_DIR, "*.png")))
print(f"Found {len(page_files)} pages to embed.")

vectors = []
for i, path in enumerate(page_files, 1):
    img = Image.open(path)
    result = vo.multimodal_embed(
        inputs=[[img]], model=MODEL, input_type="document"
    )
    vectors.append(result.embeddings[0])
    print(f"[{i}/{len(page_files)}] embedded {os.path.basename(path)}")

embeddings = np.array(vectors, dtype=np.float32)
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
with open(os.path.join(OUT_DIR, "pages.json"), "w") as f:
    json.dump([os.path.basename(p) for p in page_files], f)

print(f"\nDone. Saved {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} to {OUT_DIR}/")