'''
Created on Nov 11, 2016

@author: chiheem
'''
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import tqdm

def createData(folder_path, label_path):
    images = []
    labelNums = []
    # Cycle through the labels
    with open(label_path, 'r') as labels:
        for line in tqdm(labels,ascii=True):
            lineSplit = line.split(" ")
            filename = lineSplit[0]
            filepath = folder_path+'/'+filename
            np_image = misc.imread(filepath)
            images.append(np_image)
            labelNumber = lineSplit[1][0:-1]
            labelNums.append(labelNumber)
    return (images,labelNums)

fp = '/Users/Raoul/Dropbox (MIT)/MIT CLASSES/junior fall/6.869/FinalProject/data/images'
lp = '/Users/Raoul/Dropbox (MIT)/MIT CLASSES/junior fall/6.869/FinalProject/data/images/val.txt'
(images,labelNums) = createData(fp,lp)
print labelNums[0]
np.savez('valid.npz',images,labelNums)
