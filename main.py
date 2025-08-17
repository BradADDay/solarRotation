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
from skimage import feature, color, measure, draw
from os import listdir
from os.path import isfile, join

date = "2025-08-02"

def time2JulDate(date):
    
    #converting to julian time
    dateF = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
    julDate = sum(jdcal.gcal2jd(dateF.year, dateF.month, dateF.day))
    
    return  julDate + (dateF.hour/24) + (dateF.minute/1440) + (dateF.second/86400)

class measurement():
    
    def __init__(self, date):
        data = pd.read_csv(f"data/{date}.csv")
        
        self.imgIndex = int(data.time[0])
        self.ssCoords = data.loc[0,"radius":"xPos"].to_numpy()
        self.radius = np.mean(data.radius)
        self.error = np.sqrt(np.var(data.radius))
        self.data = data.drop(0).drop("radius", axis=1).reset_index(drop=True)
        
        self.B0 = sunpy.coordinates.sun.B0(time=date).radian
        self.L0 = sunpy.coordinates.sun.L0(time=date).radian
        self.P = sunpy.coordinates.sun.P(time=date).radian
        self.S = sunpy.coordinates.sun.angular_radius(t=date).radian
        
    def imageFitting(self, path, file, f):
            
        # Reading the image file and converting it to a numpy array of brigtness values
        image = color.rgb2gray(fits.getdata(f"{path}/{file}"), channel_axis=0)
        # Converting the pixels to be on or off
        bImage = np.float32((image > 0.1) * 1)
        
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
            self.imageFitting(path, file, f) 
        
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
        
    def coordinateRotation(self):
        # Pulling out the x,y coordinates of the suns disc
        xPos = self.data.xPos
        yPos = self.data.yPos
        ds9Err = self.error
        
        # Calculating the translation factors for each axis from the image
        xTrans = self.data.loc[self.imgIndex,"xPos"]
        yTrans = self.data.loc[self.imgIndex,"yPos"]

        # Translating the coordinates of the sun's disc and the sunspot coordinates
        xPos = xPos - xTrans
        yPos = yPos - yTrans
        ssCoords = self.ssCoords - np.array([xTrans, yTrans])

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
        
        self.sunspotV = np.array(sunspotV)
        self.sunspotVErr = np.array(sunspotVErr)
        
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
        print(chi, chiErrR)
        
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
        LL0 = np.degrees(np.arcsin(inside))

        #calculating the error in L-L0
        LL0ErrBottom = np.sqrt(1 - inside**2)
        LL0ErrB = (inside * np.tan(B) * BErr) / (LL0ErrBottom)
        LL0ErrTh = (np.sin(rho) * np.cos(chi) * (1 / np.cos(B)) * chiErr) / (LL0ErrBottom)
        LL0ErrRh = (np.cos(rho) * np.sin(chi) * (1/np.cos(B)) * rhoErr) / (LL0ErrBottom)
        LL0Err = np.degrees(np.sqrt(LL0ErrB**2 + LL0ErrTh**2 + LL0ErrRh**2))
        self.longitude = [LL0, LL0Err]
        
        self.date = time2JulDate(self.data.loc[self.imgIndex, "time"])
        
        print(self.longitude, self.latitude)
        
m0802 = measurement(date)
m0802.coordinateRotation()
m0802.heliographicConversion()