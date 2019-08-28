import requests
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import urllib
import cv2

import json

data = pd.read_json('Indian_Number_plates.json', lines=True)
pd.set_option('display.max_colwidth', -1)

del data['extras']

data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)

del data['annotation']

Images = []
Plates = []

def downloadTraining(df):

    for index, row in df.iterrows():
        resp = urllib.request.urlopen(row[0])
        im = np.array(Image.open(resp))

        Images.append(im)  

        x_point_top = row[1][0]['x']*im.shape[1]
        y_point_top = row[1][0]['y']*im.shape[0]
        x_point_bot = row[1][1]['x']*im.shape[1]
        y_point_bot = row[1][1]['y']*im.shape[0]


        carImage = Image.fromarray(im)
        plateImage = carImage.crop((x_point_top, y_point_top, x_point_bot, y_point_bot))
        Plates.append(np.array(plateImage))

downloadTraining(data)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axes
fig, ax = plt.subplots(2, 1, constrained_layout=True)

# Set title
ax[0].set_title('Input Image')
ax[1].set_title('Output Image')

# Display the images
ax[0].imshow(Images[0])
ax[1].imshow(Plates[0])

plt.show()
