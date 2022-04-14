import streamlit as st
import cv2 as cv
import tempfile
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import cv2
from collections import Counter

import os

import cv2


# %matplotlib inline
f = st.file_uploader("Upload file")
# print(f)
# tfile = tempfile.NamedTemporaryFile(delete=False)
# tfile.write(f.read())
# vf = cv.VideoCapture(tfile.name)
video_path=f.name
def get_image(image_path):
	    image = cv2.imread(image_path)
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    return image

vidcap = cv2.VideoCapture(video_path)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
x=sec/0.5

def final_func( number_of_colors):

	def RGB2HEX(color):
	    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

	


	def video_get_colors(number_of_colors):
	    
	    img_lis=[]
	    for i in range (0, int(x)):
	        image=cv2.imread("image"+str(i+1)+".jpg")
	        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	        modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
	        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

	        img_lis.append(modified_image)

	    
	    clf1 = KMeans(n_clusters = number_of_colors)
	    
	    for i in range (0, len(img_lis)):

	        clf1.fit(img_lis[i])
	    # predicting for single image 
	    labels=clf.predict(img_lis[0])
	    for i in range (0, len(img_lis)):
	        if i ==0:
	            labels=clf1.predict(img_lis[i])
	        else:
	            labels.tolist().append(clf1.predict(img_lis[i]))

	    counts = Counter(labels)
	    # sort to ensure correct color percentage
	    counts = dict(sorted(counts.items()))

	    center_colors1 = clf1.cluster_centers_
	    # We get ordered colors by iterating through the keys
	    ordered_colors1 = [center_colors1[i] for i in counts.keys()]
	    hex_colors1 = [RGB2HEX(ordered_colors1[i]) for i in counts.keys()]
	    rgb_colors1 = [ordered_colors1[i] for i in counts.keys()]


	    # plt.figure(figsize = (8, 6))
	    # plt.pie(counts.values(), labels = hex_colors1, colors = hex_colors1)
	    fig1, ax1 = plt.subplots()
	    ax1.pie(counts.values(), labels=hex_colors1, colors = hex_colors1,autopct='%1.1f%%',
	            shadow=True, startangle=90)
	    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

	    return fig1

	return video_get_colors(number_of_colors)

# text1 = st.text_area('Enter number of colors :')
number = st.number_input('Insert a number')
# k=int(float(text1))
st.pyplot(final_func(int(number)))

