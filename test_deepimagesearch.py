import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance

import os
from os import listdir
import warnings
warnings.filterwarnings("ignore")

from DeepImageSearch import Index,LoadData,SearchImage

########################################################

uploaded_file = st.file_uploader("Lade einen Kronkorken als .jpg hoch. Am besten funktioniert es wenn der Korken zugeschnittten ist!")

image_pathtest = "BierDeckel\\"+uploaded_file.name

image_list = LoadData().from_folder(folder_list = ['BierDeckel'])

Index(image_list).Start()

images = SearchImage().get_similar_images(image_path=image_pathtest,number_of_images=5)

for key, value in images.items():
    st.image(value)