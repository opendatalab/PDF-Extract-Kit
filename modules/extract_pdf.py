import os
import fitz
import numpy as np
from tqdm import tqdm
from PIL import Image

def load_pdf_fitz(pdf_path, dpi=72):
    images = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pm = page.get_pixmap(matrix=mat, alpha=False)

        # If the width or height exceeds 9000 after scaling, do not scale further.
        if pm.width > 9000 or pm.height > 9000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        images.append(np.array(img))
    return images


if __name__ == '__main__':
    for pdf in tqdm(os.listdir("data/pdfs")):
        images = load_pdf_fitz(os.path.join("data/pdfs", pdf), dpi=200)
        for idx, img in enumerate(images):
            img.save(os.path.join("data/input", pdf.replace(".pdf", f"_{idx}.jpg")))