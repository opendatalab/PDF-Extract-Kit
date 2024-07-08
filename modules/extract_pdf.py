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
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

        # if width or height > 3000 pixels, don't enlarge the image
        if pix.width > 3000 or pix.height > 3000:
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)

        # images.append(image)
        images.append(np.array(image)[:,:,::-1])
    return images


if __name__ == '__main__':
    for pdf in tqdm(os.listdir("data/pdfs")):
        images = load_pdf_fitz(os.path.join("data/pdfs", pdf), dpi=200)
        for idx, img in enumerate(images):
            img.save(os.path.join("data/input", pdf.replace(".pdf", f"_{idx}.jpg")))