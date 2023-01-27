# %%
from io import BytesIO

import streamlit as st
from PIL import Image
from rembg import remove
import glob


def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im
# %%
for imagePath in glob.glob("BierDeckel"+ "/*.jpg"):
    image = Image.open(imagePath)
    fixed = remove(image)
    
    rgb_im = fixed.convert('RGB')
    rgb_im.save(imagePath)
    


