import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from scipy import misc


def open_trainlabels():
    data = pd.read_csv('trainLabels.csv')
    return data


def preprocess_image(img):
    """This function receives an imported image and preprocess it"""
    img = img.convert('L')  # makes it greyscale
    img.show()
    y = np.asarray(img.getdata(), dtype=np.float64).reshape((img.size[1], img.size[0]))
    return 0

y = open_trainlabels()
img = Image.open('trainResized/trainResized/1.BMP', 'r')
img.show()
img = preprocess_image(img)

