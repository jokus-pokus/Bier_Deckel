# import the necessary packages
import streamlit as st
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2
import os
from rembg import remove
from PIL import Image
from io import BytesIO

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

args = "BierDeckel"
# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

uploaded_file = st.file_uploader("Lade einen Kronkorken als .jpg hoch. Am besten funktioniert es wenn der Korken zugeschnittten ist!")

if uploaded_file: 

    #save image to directory so i can open it with a path
    with open(os.path.join("BierDeckel","testfile"+uploaded_file.name),"wb") as f:
                    f.write((uploaded_file).getbuffer())

    image = Image.open("BierDeckel/"+"testfile"+uploaded_file.name)
    fixed = remove(image)
    
    rgb_im = fixed.convert('RGB')
    rgb_im.save("BierDeckel/"+"testfile"+uploaded_file.name)

    # loop over the image paths
    for imagePath in glob.glob("BierDeckel"+ "/*.jpg"):
        # extract the image filename (assumed to be unique) and
        # load the image, updating the images dictionary
        filename = imagePath[imagePath.rfind("/") + 1:]
        image = cv2.imread(imagePath)
        images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        index[filename] = hist



    testimage = "BierDeckel\\"+"testfile"+uploaded_file.name



    results = {}
    # loop over the index
    for (k, hist) in index.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = dist.cityblock(index[testimage], hist)
        results[k] = d
    # sort the results
    results = sorted([(v, k) for (k, v) in results.items()])
    results = results[:5]
    # show the query image
    st.write("Dein Bild:")
    st.image(uploaded_file)
    

    for images in results:    
        st.image(images[1])
        
        
    os.remove("BierDeckel/"+"testfile"+uploaded_file.name)
    

else: 
    st.write("Bitte ein Bild hochladen!")