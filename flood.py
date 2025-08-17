#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 20:51:45 2025

@author: brad
"""

from plottingFunction.plottingFunction import plotter
from astropy.io import fits
from skimage import feature, color, measure, draw, exposure, filters
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
import sklearn as sk
import pandas as pd

date = "2025-08-10-01"



file = "20250810_183616.fits"
# file = "20250810_183432.fits"

def binarize(image, threshold):
    # Converting the pixels to be on or off
    return np.float32(image > threshold)

class imageProcessing():
    
    def imageSequence(self, date, threshold = 0.2, path="imageSets"):
        # Finding all of the image files taken on the date
        files = [f for f in listdir(f"{path}/{date}") if isfile(join(f"{path}/{date}", f))]
        files.sort()
        dataFile = "time,radius,xPos,yPos\n"
        
        x = []
        y = []
        
        # Looping through all of the image files in the sequence, fitting the circle and pulling time information
        for file in files:
            self.file = file
            print(f"\n{file}")
            # Reading the image file and converting it to a numpy array of brigtness values
            self.image = color.rgb2gray(fits.getdata(f"{path}/{date}/{file}"), channel_axis=0)
            
            # Binarizing the image
            bImage = binarize(self.image, threshold)
            
            # Fitting the circle
            edges, circle, threshold = self.circleFitting(bImage)
            self.fitCheck(edges, circle, threshold)
            
            # Pulling the datetime information from the filenames and storing as a csv
            dataFile += file[0:4]+"-"+file[4:6]+"-"+file[6:8]+"T"+file[9:11]+":"+file[11:13]+":"+file[13:15]+f",{self.params[2]},{self.params[0]},{self.params[1]}\n"
            
            # Storing the coordinates for plotting
            x.append(self.params[1])
            y.append(self.params[0])
        
        # Writing the dates and times to a text file
        textFile = open(f"data/{date}.csv", "w")
        textFile.write(str(dataFile))
        textFile.close()
    
    def circleFitting(self, bImage, check = True, threshold=0.2):
        
        # Finding any edges within the imagez
        edges = feature.canny(bImage, sigma=5)
        coords = np.column_stack(np.nonzero(edges))
        
        # Fitting and drawing a circle to the edge coordinates
        self.model, inliers = measure.ransac(coords, measure.CircleModel, min_samples=5, residual_threshold=10)
        rr, cc = draw.circle_perimeter(int(self.model.params[0]), int(self.model.params[1]), int(self.model.params[2]), shape=bImage.shape)
        
        return [coords.T[1], coords.T[0]], [cc, rr], threshold
    
    def fitCheck(self, edges, circle, threshold):
        # Checking if the circle is larger than expected and repeating if so
        if self.model.params[2] > np.array(self.image.shape).min() / 2.5:
            print(f"Fail. Radius = {self.model.params[2]}")
            self.threshold += 0.02
            self.circleFitting(self.image, threshold) 
        
        else:
            # Showing the image with the fit circle
            plot = plotter(title=f"{self.file[:-6]}")
            plot.imshow(self.image)
            plot.scatter([self.model.params[1], self.model.params[0]], s=50, c='red', marker="x")
            plot.scatter(edges, s=0.05)
            plot.scatter(circle, s=1, c="r")
            plot.aspect()
            plot.invert("y")
            plt.show()
            
            print("Success")
            
            self.params = self.model.params
    
    def sunspotLocator(self, file, date, path = "imageSets", high=9, low=8, threshold=0.9):
        
        # Reading the image file and converting it to a numpy array of brigtness values
        self.image = color.rgb2gray(fits.getdata(f"{path}/{date}/{file}"), channel_axis=0)
        
        # Increasing the contrast and binarizing the image, then fitting a circle to crop it
        image, radius, bounds = self.contrast(self.image, high, low, threshold)
        x = []
        y = []
        
        # Storing the dark pixels as coordinates
        for i in range(len(image)):
            print(i)
            for j in range(len(image)):
                if image[i][j] == 0:
                    x.append(j)
                    y.append(i)
        
        # Translating the coordinates to get their positions on the uncropped image
        binaryX = np.array(x) + bounds[0]
        binaryY = np.array(y) + bounds[2]
        data = np.array([binaryX, binaryY])
        
        # Applying gaussian mixture to cluster the data and isolate each sunspot
        clusters = int(input("How many sunspots are there?: "))
        clusteredX, clusteredY, uniqueIndices, clusterIndices = self.gaussMix(data.T, clusters)
        
        ssCoords = []
        
        # Taking the centre of mass of each sunspot and storing to an array
        for i in range(len(clusteredX)):
            sunspotX = np.array(clusteredX[i]).mean()
            sunspotY = np.array(clusteredY[i]).mean()
            ssCoords.append([sunspotX, sunspotY, int(file[9:15]), int(uniqueIndices[i])])
        ssCoords = np.array(ssCoords)
        
        # Plotting the final coordinates and saving as a csv
        plot = self.plotting(ssCoords, bounds)
        plot.save(f"data/{date}_ss.png")
        ssCoords = pd.DataFrame(ssCoords[:,0:3], index=ssCoords[:,3].astype(int), columns = ["x","y","t"])
        
        ssCoords.to_csv(f"data/{file[:-5]}_ss.csv")
        
        return ssCoords
            
    def plotting(self, ssCoords, bounds):
        # Showing the image with the fitting parameters
        plot = plotter(figsize=(16,12))
        plot.imshow(self.image)
        plot.aspect()
        for i in range(len(ssCoords)):
            coords = ssCoords[i]
            print(coords)
            plot.ax.annotate(f'{int(coords[3])}', xy=coords[0:2], textcoords='data')
            plot.scatter(coords[0:2], s=20, label = int(coords[3]))
        plot.limits(bounds[0],bounds[1],bounds[2],bounds[3])
        plot.invert("y")
        plt.show()
        
        return plot
        
    def contrast(self, image, high, low, threshold):
        
        # Filtering to find regions with a fast change in intensity and binarizing
        image = filters.difference_of_gaussians(image, low, high)
        image = exposure.rescale_intensity(image, (image.min(), image.max()), (0,1))
        image = binarize(image, threshold*image.mean())
        
        edges, circle, threshold = self.circleFitting(image, False)
        
        return self.cropping(image)
        
    def cropping(self, image):
        
        radius = int(self.model.params[2])
    
        boundsX = [int(self.model.params[1] - radius), int(self.model.params[1] + radius)]
        boundsY = [int(self.model.params[0] - radius), int(self.model.params[0] + radius)]
        bounds = [boundsX[0], boundsX[1], boundsY[0], boundsY[1]]
    
        image = image[boundsY[0]:boundsY[1],boundsX[0]:boundsX[1]]
    
        for i in range(len(image)):
            for j in range(len(image)):
                if (i - radius)**2 + (j - radius)**2 > (0.95*radius)**2:
                    image[i][j] = 1
        
        plot = plotter()
        plot.imshow(image)
        plt.show()
                    
        return image, radius, bounds
    
    def gaussMix(self, dataFrame, clusters):

        # Clustering
        gaussMix = sk.mixture.GaussianMixture(clusters, random_state=0).fit(dataFrame)
        # Taking the Indices and creating a list of the unique values
        clusterIndices = gaussMix.predict(dataFrame)
        uniqueIndices = np.unique(clusterIndices)
        
        # Creating lists for storage
        clusteredX = []
        clusteredY = []
        
        # Grouping the data in 2D arrays based on its cluster
        for cluster in uniqueIndices:
            clusteredX.append(dataFrame[np.where(clusterIndices == cluster),0])
            clusteredY.append(dataFrame[np.where(clusterIndices == cluster),1])
        
        return clusteredX, clusteredY, uniqueIndices, clusterIndices
                    
dates = [f for f in listdir("imageSets") if isdir(join("imageSets", f))]
dates.sort()

files = ["20250802_094414.fits", "20250804_155723.fits", "20250805_173427.fits",
         "20250809_140702.fits", "20250809_192412.fits", "20250810_102426.fits",
         "20250810_183616.fits"]

for i in range(len(files)):
    imageProcessing().sunspotLocator(files[i], dates[i])
    
    
    