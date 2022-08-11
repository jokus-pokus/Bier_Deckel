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



def extract(file):
  file = Image.open(file).convert('L').resize(IMAGE_SHAPE)

  file = np.stack((file,)*3, axis=-1)

  file = np.array(file)/255.0

  embedding = model.predict(file[np.newaxis, ...])

  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()

  return flattended_feature


uploaded_file = st.file_uploader("Choose a file")

st.image(uploaded_file)

#breakpoint() #streamlit breakpoint
# model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

# IMAGE_SHAPE = (224, 224)

# layer = hub.KerasLayer(model_url)
# model = tf.keras.Sequential([layer])


# df = pd.DataFrame(columns=['Bild', 'Vektor'])

# #Process all files in folder
# folder_dir = "D:/Python/Bier_Deckel/BierDeckel"
# for images in os.listdir(folder_dir):
 
#     # check if the image ends with png
#     if (images.endswith(".jpg")):
#         vector = extract(folder_dir+'/'+images)
#         #print(vector.size)
#         new_row = {'Bild':images, 'Vektor':vector}
#         df = df.append(new_row,ignore_index=True)
#         #print(images)

# #Process the uploaded file
# vector =  extract(uploaded_file)
# new_row = {'Bild':'newImage', 'Vektor':vector}
# df = df.append(new_row,ignore_index=True)

# testinput = df['Vektor'].iloc[-1] 

# #Calc distances
# PicData = pd.DataFrame(columns=['Bild', 'Dist'])
# for ind in df.index:
#     dc = distance.cdist([testinput], [df.Vektor[ind]], metric='cosine')[0] #cosine
#     #print(dc)
#     new_row = {'Bild':df.Bild[ind], 'Dist':dc}
#     PicData = PicData.append(new_row,ignore_index=True)
# PicData.Dist = PicData.Dist.astype('float32')

# PrintData = PicData.sort_values(by=['Dist']).head(10)


# #Über alle Bilder itterieren

# for Bild , Dist in PrintData.itertuples(index=False):
#     plt.rcParams["figure.figsize"] = [2, 2]
#     plt.rcParams["figure.autolayout"] = True
#     im = plt.imread("D:/Python/Bier_Deckel/BierDeckel/"+Bild)
#     fig, ax = plt.subplots()
#     fig.suptitle(np.round_(Dist,decimals=3))
#     im = ax.imshow(im, extent=[0, 300, 0, 300])
#     plt.show()
