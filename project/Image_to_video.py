# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:43:46 2019

@author: abder
"""
import cv2
import glob

file_loc=str(input("Please give the direction of your file"))
img_array = []
for filename in glob.glob('*.tiff'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter(file_loc,cv2.VideoWriter_fourcc(*'DIVX'), 4, (width,height))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()