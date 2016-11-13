'''
Created on Nov 11, 2016

@author: chiheem
'''
import os
import re
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time

def createData(folder_path, label_path):
    '''
    classdocs
    '''
    # Initialize dictionary
    image_label = []
    label_mapping = []
    label_int = 0
    
    # Cycle through the labels
    label_file = open(label_path)
    for line in label_file:
        #Clean label name
        label_name = re.sub('[^A-Za-z0-9_/.]+', "", line)
        # Add label and int into label_mapping
        label_mapping.append([label_int, label_name])
        #Get all the files in folder
        currentPath = folder_path+'/'+label_name
        filenames = [f for f in os.listdir(currentPath) if os.path.isfile(os.path.join(currentPath, f))]
        #For each file...
        for filename in filenames:
            #Create path to file
            filepath = currentPath+'/'+filename
            #Open image as np array
            np_image = misc.imread(filepath)
            image_label.append([np_image,label_int])
        #Update label integer
        label_int+=1
    return image_label
    
"""
folder_path = '/home/chiheem/workspace/miniScene/images/train'
label_path = '/home/chiheem/workspace/miniScene/images/labels.txt'
data=createData(folder_path, label_path)
print(data)
"""
