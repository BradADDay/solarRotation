#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 15:43:36 2025

@author: brad
"""

import sunpy.coordinates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import jdcal
import datetime
from plottingFunction.plottingFunction import plotter
from astropy.io import fits
from skimage import feature, color, measure, draw, filters, exposure
from os import listdir
from os.path import isfile, join

def time2JulDate(date):
    
    #converting to julian time
    dateF = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
    julDate = sum(jdcal.gcal2jd(dateF.year, dateF.month, dateF.day))
    
    return  julDate + (dateF.hour/24) + (dateF.minute/1440) + (dateF.second/86400)

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
            
            # Fitting the circle
            self.circleFitting(self.image)
            
            # Pulling the datetime information from the filenames and storing as a csv
            dataFile += file[0:4]+"-"+file[4:6]+"-"+file[6:8]+"T"+file[9:11]+":"+file[11:13]+":"+file[13:15]+f",{self.params[2]},{self.params[0]},{self.params[1]}\n"
            
            # Storing the coordinates for plotting
            x.append(self.params[1])
            y.append(self.params[0])
        
        # Writing the dates and times to a text file
        textFile = open(f"data/{date}.csv", "w")
        textFile.write(str(dataFile))
        textFile.close()
        
    def binarize(self, image, threshold):
        # Converting the pixels to be on or off
        return np.float32(image > threshold)
    
    def circleFitting(self, image, threshold=0.2):
        self.threshold = threshold
        
        bImage = self.binarize(image, threshold)
        
        # Finding any edges within the imagez
        edges = feature.canny(bImage, sigma=5)
        coords = np.column_stack(np.nonzero(edges))
        
        # Fitting and drawing a circle to the edge coordinates
        self.model, inliers = measure.ransac(coords, measure.CircleModel, min_samples=5, residual_threshold=10)
        rr, cc = draw.circle_perimeter(int(self.model.params[0]), int(self.model.params[1]), int(self.model.params[2]), shape=bImage.shape)
        
        self.fitCheck([coords.T[1], coords.T[0]], [cc, rr], threshold)
    
    def fitCheck(self, edges, circle, threshold):
        # Checking if the circle is larger than expected and repeating if so
        if self.model.params[2] > np.array(self.image.shape).min() / 2.5:
            print(f"Fail. Radius = {self.model.params[2]}")
            self.threshold += 0.02
            self.circleFitting(self.image, threshold) 
        
        else:
            # Showing the image with the fit
            plot = plotter(title=f"{self.file[:-6]}")
            plot.imshow(self.image)
            plot.scatter([self.model.params[1], self.model.params[0]], s=50, c='red', marker="x")
            plot.scatter(edges, s=0.05)
            plot.scatter(circle, s=1, c="r")
            plot.aspect()
            plt.show()
            
            print("Success")
            
            self.params = self.model.params

class measurement():
    
    def analyse(self, date, sunspot):
        # Reading the file containing sunspot positions
        self.sunspotDF = pd.read_csv(f"data/{sunspot}.csv", index_col=0).loc[date]
        
        # Pulling heliographic values
        self.B0 = sunpy.coordinates.sun.B0(time=date).radian
        self.L0 = sunpy.coordinates.sun.L0(time=date).radian
        self.P = sunpy.coordinates.sun.P(time=date).radian
        self.S = sunpy.coordinates.sun.angular_radius(t=date).radian
        
        # Checking which function to call for dataset type
        if self.sunspotDF.type == "T":
            self.coordinateRotation()
            return self.latitude, self.longitude, self.date
        elif self.sunspotDF.type == "A":
            self.SDO()
            return self.latitude, self.longitude, self.date
        else:
            print("Invalid Entry")
        
    def SDO(self):
        # Setting the North Vector
        self.northMag = 1
        self.northMagErr = 0.0001
        self.sunNorthV = np.array([0,1])
        
        # Setting the centre coordinates of the sun's disc for translation
        xTrans = 513
        yTrans = 513
        
        # Translating the sunspot coordinates
        self.sunspotV =  np.array([self.sunspotDF.x - xTrans, self.sunspotDF.y - yTrans])
        
        # Error in sunspot array
        self.sunspotVErr = np.array([2, 2])
        self.error = 2
        
        # Setting the sun's radius
        self.radius = 395
        
        # Pulling the julian date
        self.date = time2JulDate(self.sunspotDF.name)
        
        # Converting to heliographic coordinates
        self.heliographicConversion()
        
    def coordinateRotation(self):
        
        # Pulling information from the datafile
        imgIndex = self.sunspotDF.loc["imgIndex"]
        ssCoords = self.sunspotDF.loc[["x","y"]].to_numpy()
        data = pd.read_csv(f"data/{date}.csv")
        
        self.radius = np.mean(data.radius)
        self.error = np.sqrt(np.var(data.radius))
        
        # Pulling out the x,y coordinates of the suns disc
        xPos = data.xPos
        yPos = data.yPos
        ds9Err = self.error
        
        # Calculating the translation factors for each axis from the image
        xTrans = data.loc[imgIndex,"xPos"]
        yTrans = data.loc[imgIndex,"yPos"]

        # Translating the coordinates of the sun's disc and the sunspot coordinates
        xPos = xPos - xTrans
        yPos = yPos - yTrans
        ssCoords = ssCoords - np.array([xTrans, yTrans])

        # Computing a linear fit to the data to find the slope
        linFit = linregress(xPos, yPos)
        motionSlope = linFit.slope
        motionSlopeErr = linFit.stderr
        motionIntercept = 0
        motionInterceptErr = np.sqrt((motionSlope * ds9Err)**2 + ds9Err**2)
        
        # Plotting the motion of the sun across the field of view
        plot = plotter()
        plot.defaultScatter([xPos, yPos], ["x Position", "y Position"])
        plot.plot([xPos, xPos*motionSlope+motionIntercept])

        # Calculating a vector pointing north, perpendicular to the apparent motion of the sun
        northSlope, northSlopeErr = (-1/motionSlope), ((1/motionSlope**2)*motionSlopeErr)
        northV = [500, northSlope * 500 + motionIntercept]
        northVErr = [ds9Err, np.sqrt((500*northSlopeErr)**2 + motionInterceptErr**2 + ds9Err**2)]
        
        # Calculating the magnitude of the north vector and its error
        northMag = np.linalg.norm(northV)
        northMagErr1 = northV[0] * northVErr[1] / northMag
        northMagErr2 = northV[1] * northVErr[1] / northMag
        northMagErr = np.linalg.norm([northMagErr1, northMagErr2])
        
        # Settomg attributes to be used for heliographic coordinate conversion
        self.sunNorthV = np.array([0, northMag])
        self.sunNorthVErr = np.array([0, northMagErr])
        self.northMag = northMag
        self.northMagErr = northMagErr
        
        # Calculating the angle between the north vector and the vertical axis
        psi = np.arccos(northV[1] / northMag)
        psiErrSqrt = np.sqrt(1 - (northV[1]/northMag)**2)
        psiErr = np.sqrt(((northV[1]*northMagErr)/(psiErrSqrt*(northMag**2)))**2 + (northVErr[1]/(psiErrSqrt*northMag))**2)
        
        # Adjusting the angle if it lies on the left side of the y axis
        if northV[0] <= 0:
            psi = 2*np.pi - psi
        
        # Rotating such that the rotational axis of the sun is vertical
        phi = psi - self.P
        self.phi = phi
        
        # Rotating the sunspot vector in line with the coordinate system
        sunspotV = [ssCoords[0]*np.cos(phi)-ssCoords[1]*np.sin(phi), ssCoords[0]*np.sin(phi)+ssCoords[1]*np.cos(phi)]
        sunspotVEx = (ds9Err*np.cos(phi))**2 + (ds9Err*np.sin(phi))**2 + ((sunspotV[0]*np.sin(phi) - sunspotV[1]*np.cos(phi))*psiErr)**2
        sunspotVEy = (ds9Err*np.cos(phi))**2 + (ds9Err*np.sin(phi))**2 + ((sunspotV[0]*np.cos(phi) - sunspotV[1]*np.sin(phi))*psiErr)**2
        sunspotVErr = [np.sqrt(sunspotVEx), np.sqrt(sunspotVEy)]
        
        # Setting attributes to be used for heliographic conversion
        self.sunspotV = np.array(sunspotV)
        self.sunspotVErr = np.array(sunspotVErr)
        
        # Saving the date to return
        self.date = time2JulDate(data.loc[imgIndex, "time"])
        
        self.heliographicConversion()
        
    def heliographicConversion(self):
        
        sunspotV = self.sunspotV
        sunspotVErr = self.sunspotVErr
        sunNorthV = self.sunNorthV
        
        # Finding the magnitude of the sunspot vector
        ssVectMag = np.linalg.norm(sunspotV)
        ssVectMagErr = np.linalg.norm(sunspotV * sunspotVErr / ssVectMag)
        
        rho = np.sin(ssVectMag / self.radius) - (ssVectMag * self.S) / self.radius
        
        rhoErr1 = ((np.cos(ssVectMagErr / self.radius) - self.S) / self.radius)*ssVectMagErr
        rhoErr2 = ((self.S * ssVectMag - np.cos(ssVectMag / self.radius) * ssVectMag) / self.radius**2) * self.error
        rhoErr = np.linalg.norm([rhoErr1, rhoErr2])
        
        top = np.dot(sunspotV, sunNorthV)
        bottom = (ssVectMag * np.linalg.norm(sunNorthV))
        mChi = np.arccos(top / bottom)
        
        if sunspotV[0] >= 0:
            mChi = 2*np.pi - mChi
            
        chi = -mChi
        
        bottomErr = self.northMag * ssVectMag * np.sqrt(1 - (top / bottom)**2)
        
        chiErrS = sunNorthV * sunspotVErr / bottomErr
        chiErrNy = (sunNorthV[1]*sunspotV[1])/(bottomErr)
        chiErrR = (top*ssVectMagErr)/(bottomErr*ssVectMag)
        chiErrN = (top*self.northMagErr)/(bottomErr*self.northMag)
        
        chiErr = np.sqrt(chiErrS[0]**2 + chiErrS[1]**2 + chiErrNy**2 + chiErrR**2 + chiErrN**2)
        
# =============================================================================
#         Calculating heliographic coordinates
# =============================================================================
        
        #finding B
        B1 = np.sin(self.B0)*np.cos(rho)
        B2 = np.cos(self.B0)*np.sin(rho)*np.cos(chi)
        B = np.arcsin(B1+B2)
        
        #calculating the error in B
        BErrBottom = np.sqrt(1 - (np.cos(self.B0)*np.sin(rho)*np.cos(mChi) + np.sin(self.B0)*np.cos(rho))**2)
        BErrChi = (np.cos(self.B0)*np.sin(rho)*np.cos(mChi)*chiErr)/BErrBottom
        BErrRho = ((np.cos(self.B0)*np.cos(rho)*np.cos(mChi) - np.sin(self.B0)*np.sin(rho))*rhoErr)/BErrBottom
        BErr = np.sqrt(BErrChi**2 + BErrRho**2)
        self.latitude = [B, BErr]
        
        #calculating L-L0
        inside = np.sin(rho) * np.sin(chi) * (1/np.cos(B))
        LL0 = np.rad2deg(np.arcsin(inside))

        #calculating the error in L-L0
        LL0ErrBottom = np.sqrt(1 - inside**2)
        LL0ErrB = (inside * np.tan(B) * BErr) / (LL0ErrBottom)
        LL0ErrTh = (np.sin(rho) * np.cos(chi) * (1 / np.cos(B)) * chiErr) / (LL0ErrBottom)
        LL0ErrRh = (np.cos(rho) * np.sin(chi) * (1/np.cos(B)) * rhoErr) / (LL0ErrBottom)
        LL0Err = np.rad2deg(np.sqrt(LL0ErrB**2 + LL0ErrTh**2 + LL0ErrRh**2))
        self.longitude = [LL0, LL0Err]
        
        print(self.longitude[0], self.latitude[0])


sunspot = "SDO/SDO_2024-06-05_2024-06-10"
# sunspot = "sunspotCoords"

df = pd.read_csv(f"data/{sunspot}.csv", index_col=0)
dates = df.index

lats = []
latsE = []
lons = []
lonsE = []
dats = []

for date in dates:
    lat, lon, dat = measurement().analyse(date, sunspot)
    lats.append(lat[0])
    latsE.append(lat[1])
    lons.append(lon[0])
    lonsE.append(lon[1])
    dats.append(dat)
    
latplot = plotter(title=sunspot[4:])
latplot.defaultScatter([dats, lats], ["Julian Date", "Latitude"])
latplot.errorbar([dats, lats], [None, latsE])
lonplot = plotter(title=sunspot[4:])
lonplot.defaultScatter([dats, lons], ["Julian Date", "Longitude"])
lonplot.errorbar([dats, lons], [None, lonsE])