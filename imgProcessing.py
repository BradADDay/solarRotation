#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:54:52 2025

@author: brad
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import feature, color, measure, draw
from os import listdir
from os.path import isfile, join
from plottingFunction.plottingFunction import plotter

def fitting(path, file, f):
        
    # Reading the image file and converting it to a numpy array of brigtness values
    image = color.rgb2gray(fits.getdata(f"{path}/{file}"), channel_axis=0)
    # Converting the pixels to be on or off
    bImage = np.float32((image > 0.2) * 1)
    
    # Finding any edges within the image
    edges = feature.canny(bImage, sigma=5)
    coords = np.column_stack(np.nonzero(edges))
    
    # Taking the xy coordinates of any edges for plotting
    edgesX = coords.T[0]
    edgesY = coords.T[1]
    
    # Fitting and drawing a circle to the edge coordinates
    model, inliers = measure.ransac(coords, measure.CircleModel, min_samples=5, residual_threshold=10)
    rr, cc = draw.circle_perimeter(int(model.params[0]), int(model.params[1]), int(model.params[2]), shape=image.shape)
    
    # Checking if the circle is larger than expected and repeating if so
    if model.params[2] > np.array(image.shape).min() / 2.5:
        print(f"Fail. Radius = {model.params[2]}")
        fitting(path, file, f) 
    
    else:
        # Pulling the datetime information from the filenames and storing as a csv
        date = file[:8]
        time = file[9:15]
        f += date[0:4]+"-"+date[4:6]+"-"+date[6:8]+"T"+time[0:2]+":"+time[2:4]+":"+time[4:6]+f",{model.params[2]},{model.params[0]},{model.params[1]}\n"
        
        # Showing the image with the fitting parameters
        plot = plotter()
        plot.imshow(image)
        plot.scatter([model.params[1], model.params[0]], s=50, c='red', marker="x")
        plot.scatter([edgesY, edgesX], s=0.05)
        plot.scatter([cc, rr], s=1, c="r")
        plot.aspect()
        plt.show()
        
        print(f"\n{file}")
        print(model.params)
        
        return f, model.params

# The date of the image sequence being taken
date = "2025-08-04"
path = "imageSets"

# Finding all of the image files taken on the date
files = [f for f in listdir(f"{path}/{date}") if isfile(join(f"{path}/{date}", f))]
files.sort()
dataFile = "time,radius,xPos,yPos\n"


x = []
y = []

for file in files:
    dataFile, params = fitting(f"imageSets/{date}", file, dataFile)
    
    # Storing the coordinates for plotting
    x.append(params[1])
    y.append(params[0])
    
#writing the dates and times to a text file
textFile = open(f"data/{date}.csv", "w")
textFile.write(str(dataFile))
textFile.close()

plot = plotter()
plot.defaultScatter([x,y], ["x", "y"])
plot.limits(0,4000,0,3000)
plot.aspect()
plot.invert("y")
