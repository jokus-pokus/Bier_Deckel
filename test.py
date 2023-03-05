import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
import io
import os
from os import listdir
import warnings
warnings.filterwarnings("ignore")


@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

    layer = hub.KerasLayer(model_url)
    model = tf.keras.Sequential([layer])
    return model

model = load_model()

def extract(file):
    IMAGE_SHAPE = (224, 224)

    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)

    file = np.stack((file,)*3, axis=-1)

    file = np.array(file)/255.0

    embedding = model.predict(file[np.newaxis, ...])

    vgg16_feature_np = np.array(embedding)
    flattended_feature = vgg16_feature_np.flatten()

    return flattended_feature


uploaded_file = st.file_uploader("Lade einen Kronkorken als .jpg hoch. Am besten funktioniert es wenn der Korken zugeschnittten ist!")

if uploaded_file: 

    #save image to directory so i can open it with a path
    with open(os.path.join("BierDeckel","testfile"+uploaded_file.name),"wb") as f:
                    f.write((uploaded_file).getbuffer())

    st.write("Dein Bild:")
    st.image(uploaded_file)

    df = pd.DataFrame(columns=['Bild', 'Vektor'])

    #Process all files in folder
    folder_dir = "BierDeckel"
    for images in os.listdir(folder_dir):

        # check if the image ends with png
        if (images.endswith(".jpg")):
            vector = extract(folder_dir+'/'+images)
            #print(vector.size)
            new_row = {'Bild':images, 'Vektor':vector}
            df = df.append(new_row,ignore_index=True)


    testinput = df['Vektor'].iloc[-1] 

    #Calc distances
    PicData = pd.DataFrame(columns=['Bild', 'Dist'])
    for ind in df.index:
        dc = distance.cdist([testinput], [df.Vektor[ind]], metric='cosine')[0] #cosine
        #print(dc)
        new_row = {'Bild':df.Bild[ind], 'Dist':dc}
        PicData = PicData.append(new_row,ignore_index=True)
    PicData.Dist = PicData.Dist.astype('float32')

    PrintData = PicData.sort_values(by=['Dist']).head(6)
    PrintData = PrintData.iloc[1:]


    # #Über alle Bilder itterieren
    st.write("Die möglichen Doppel:")
    for Bild , Dist in PrintData.itertuples(index=False):
        im = plt.imread("BierDeckel/"+Bild)
        st.image(im)

        #Aufräumen
    del(df)
    del(PrintData)
    del(PicData)
    del(testinput)
    #os.remove("BierDeckel/"+"testfile"+uploaded_file.name) 
    #del(uploaded_file)

    #Neues Bild in der DB Speichern?    
    if st.button('Neues Bild in der DB Speichern?'):
        st.write("Bild wurde gespeichert!")
    else:
        st.write("Bild wurde nicht gespeichert!")
        os.remove("BierDeckel/"+"testfile"+uploaded_file.name)
        del(uploaded_file)

else: 
        st.write("Bitte ein Bild hochladen!")

