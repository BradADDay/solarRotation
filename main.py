"""
Created on Tue Aug  5 15:43:36 2025

@author: brad
"""

import sunpy.coordinates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import jdcal
import datetime
from plottingFunction.plottingFunction import plotter
from astropy.io import fits
from skimage import feature, color, measure, draw, filters, exposure
from os import listdir
from os.path import isfile, join
import sklearn as sk

# =============================================================================
# Misc Functions
# =============================================================================

def time2JulDate(date):
    """Converts a date string in the form YY-mm-ddTHH:MM:SS to julian day"""
    # Converting to datetime
    dateF = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
    # Converting to julian date
    julDate = sum(jdcal.gcal2jd(dateF.year, dateF.month, dateF.day))
    return  julDate + (dateF.hour/24) + (dateF.minute/1440) + (dateF.second/86400)

def binarize(image, threshold):
    """Makes a binary image based on a brightness threshold"""
    return np.float32(image > threshold)

def weightedMean(coords, weights):
    """Takes the weighted mean of coordinates"""
    weightedCoords = []
    weight = []
    
    for coord in coords:
        weightedCoords.append(coord * weights[coord[0]][coord[1]])
        weight.append(weights[coord[0]][coord[1]])
    weightedCoords = np.array(weightedCoords)
        
    xMean = np.sum(weightedCoords[:,0])/np.sum(weight)
    yMean = np.sum(weightedCoords[:,1])/np.sum(weight)
    xE = np.sqrt(np.sum(coords[:,0]-xMean) **2 / len(coords)) / np.sqrt(len(coords))
    yE = np.sqrt(np.sum(coords[:,1]-yMean) **2 / len(coords)) / np.sqrt(len(coords))
    
    return xMean, yMean, xE, yE

def isAdjacent(a1, a2):
    """Checks if a1 and a2 have any adjacent pixels"""
    a1 = np.array(a1).T
    adjacent=False
    i=0
    
    while i <= len(a1[0])-1:
        x,y = a1[0][i], a1[1][i]
        
        if a2[(a2 == (x+1,y)).all(axis=1)].size>0:
            adjacent = True
        elif a2[(a2 == (x-1,y)).all(axis=1)].size>0:
            adjacent = True
        elif a2[(a2 == (x,y+1)).all(axis=1)].size>0:
            adjacent = True
        elif a2[(a2 == (x,y-1)).all(axis=1)].size>0:
            adjacent = True
            
        if adjacent == True:
            break
        
        i += 1
    
    return adjacent
    
def lin(x, m, c):
    return m * x + c

# =============================================================================
# Image Processing
# =============================================================================

class imageProcessing():
    
    def imageSequence(self, date, threshold = 0.2, path="../imageSets", radBounds = [950,1150]):
        """Initialises the analysis of a sequence of images"""
        # Finding all of the image file names taken on the date
        files = [f for f in listdir(f"{path}/{date}") if isfile(join(f"{path}/{date}", f))]
        files.sort()
        dataFile = "time,radius,xPos,yPos\n"
        
        self.i = 0
        
        # Looping through all of the image files in the sequence, fitting the circle and pulling time information
        for file in files:
            self.file = file
            print(f"\n{file}")
            
            # Reading the image file and converting it to a numpy array of brigtness values
            self.image = color.rgb2gray(fits.getdata(f"{path}/{date}/{file}"), channel_axis=0)
            
            # Binarizing the image
            bImage = binarize(self.image, threshold)
            
            # Fitting the circle
            self.circleFitting(bImage, radBounds, threshold)
            
            # Pulling the datetime information from the filenames and storing as a csv
            dataFile += file[0:4]+"-"+file[4:6]+"-"+file[6:8]+"T"+file[9:11]+":"+file[11:13]+":"+file[13:15]+f",{self.model.params[2]},{self.model.params[1]},{self.model.params[0]}\n"
            self.i += 1
        
        # Checking what image to use for sunspot location and calling the location function
        fileIndex = int(input("Which file will be used for sunspots?: "))
        self.telData(files[fileIndex], date)
        
        # Writing the dates and times to a text file
        textFile = open(f"data/{files[fileIndex][:-5]}.csv", "w")
        textFile.write(str(dataFile))
        textFile.close()
    
    def circleFitting(self, bImage, radBounds=None, threshold=0.2, check = True):
        """Fits a circle to the binarized image to find the centre and radius of the disc"""
        # Finding any edges within the imagez
        edges = feature.canny(bImage, sigma=5)
        coords = np.column_stack(np.nonzero(edges))
        
        # Fitting and drawing a circle to the edge coordinates
        self.model, inliers = measure.ransac(coords, measure.CircleModel, min_samples=5, residual_threshold=10)
        rr, cc = draw.circle_perimeter(int(self.model.params[0]), int(self.model.params[1]), int(self.model.params[2]), shape=bImage.shape)
        
        # Checking if the fit is within expectationbounds
        if check == True:
            self.fitCheck(bImage, [coords.T[1], coords.T[0]], [cc, rr], threshold, radBounds)
        else:
            print("Success")
            return rr, cc
    
    def fitCheck(self, bImage, edges, circle, threshold, radBounds):
        """Checks the circular fit to the data"""
        # Checking if the circle is larger than expected and repeating if so
        if (self.model.params[2] <= radBounds[0] or self.model.params[2] >= radBounds[1]):
            
            # If the radius is larger than the bounds, increases the threshold and repeats the fit
            threshold += 0.02
            print(f"Fail. Radius = {self.model.params[2]}, increasing threshold to {threshold}")
            self.circleFitting(bImage, radBounds, threshold) 
        
        else:
            # Showing the image with the fit circle
            plot = plotter(title=f"img_{self.i}: {self.file[:-5]}")
            plot.imshow(self.image)
            plot.scatter([self.model.params[1], self.model.params[0]], s=50, c='red', marker="x")
            plot.scatter(edges, s=0.05)
            plot.scatter(circle, s=1, c="r")
            plot.aspect()
            plot.invert("y")
            plt.show()
            
            print("Success")
    
    def sdoData(self, subfolder, path = "../imageSets/SDO", high=15, low=5, threshold=0.9, clusters = 50):
        
        # Finding all of the image file names taken on the date
        files = [f for f in listdir(f"{path}/{subfolder}") if isfile(join(f"{path}/{subfolder}", f))]
        files.sort()
        
        for file in files:
            print(f"{file}...")
            self.image = fits.getdata(f"{path}/{subfolder}/{file}")
            
            if self.image.shape[0] == 3:
                self.image = color.rgb2gray(self.image, channel_axis=0)
                
            self.image = exposure.rescale_intensity(self.image, (self.image.min(), self.image.max()), (0,1))
            
            ssCoords, bounds = self.sunspotLocator(high, low, threshold, clusters)
            
            filename = f"{file[:-15]}_ss"
            dataFile = "time,radius,xPos,yPos\n"
            dataFile += file[0:4]+"-"+file[4:6]+"-"+file[6:8]+"T"+file[9:11]+":"+file[11:13]+":"+file[13:15]+f",{self.model.params[2]},{self.model.params[1]},{self.model.params[0]}\n"
            
            # Writing the dates and times to a text file
            textFile = open(f"data/{file[:-15]}.csv", "w")
            textFile.write(str(dataFile))
            textFile.close()
            
            self.plotting(ssCoords, bounds, filename)
            ssCoords = pd.DataFrame(ssCoords, index=ssCoords[:,2].astype(int), columns = ["x","y","t","xE","yE"])
            ssCoords.to_csv(f"data/{filename}.csv")
    
    def telData(self, file, subfolder, path = "../imageSets", high=9, low=8, threshold=0.9, clusters = 50):
        """Uses image processing routines to increase the contrast of sunspots and pull their coordinates"""
        # Reading the image file and converting it to a numpy array of brigtness values
        self.image = color.rgb2gray(fits.getdata(f"{path}/{subfolder}/{file}"), channel_axis=0)
        self.image = exposure.rescale_intensity(self.image, (self.image.min(), self.image.max()), (0,1))
        
        ssCoords, bounds = self.sunspotLocator(high, low, threshold, clusters)
        
        # Plotting the final coordinates and saving the data as a csv
        filename = f"{file[:-5]}_ss"
        self.plotting(ssCoords, bounds, filename)
        ssCoords = pd.DataFrame(ssCoords, index=ssCoords[:,2].astype(int), columns = ["x","y","t","xE","yE"])
        ssCoords.to_csv(f"data/{filename}.csv")

    def sunspotLocator(self, high, low, threshold, clusters):
        # Increasing the contrast and binarizing the image, then fitting a circle to crop it
        image, radius, bounds = self.contrast(self.image, high, low, threshold)
        x = []
        y = []
        
        # Storing the dark pixels as coordinates
        for i in range(len(image)):
            for j in range(len(image)):
                if image[i][j] == 0:
                    x.append(j)
                    y.append(i)
        
        # Translating the coordinates to get their positions on the uncropped image
        binaryX = np.array(x) + bounds[0]
        binaryY = np.array(y) + bounds[2]
        data = np.array([binaryX, binaryY])
        
        return self.clusteringCoM(data, clusters), bounds

    def clusteringCoM(self, data, clusters):
        
        # Applying gaussian mixture to cluster the data and isolate each sunspot
        clusteredX, clusteredY, uniqueIndices = self.gaussMix(data.T, clusters)
        ssCoords = []
        
        # Taking the weighted centre of mass of each sunspot and storing to an array
        for i in range(len(clusteredX)):
            coords = np.array((clusteredX[i], clusteredY[i])).T
            
            # Taking the weighted mean of the pixels weighing by their darkness values
            sunspotX, sunspotY, sunspotXE, sunspotYE = weightedMean(coords, 1-self.image)
            ssCoords.append([sunspotX, sunspotY, int(uniqueIndices[i]), sunspotXE, sunspotYE])
        
        return np.array(ssCoords)
        
    def contrast(self, image, high, low, threshold):
        """Increasing the contrast of the image"""
        # Fitting to get a circle perimeter for plotting and data analysis
        self.cc, self.rr = self.circleFitting(binarize(image, 0.2), check = False)
        
        # Filtering to find regions with a fast change in intensity and binarizing
        image = filters.difference_of_gaussians(image, low, high)
        plot = plotter()
        plot.imshow(image)
        image = exposure.rescale_intensity(image, (image.min(), image.max()), (0,1))
        image = binarize(image, threshold*image.mean())
        
        plot = plotter()
        plot.imshow(image)
        
        return self.cropping(image)
        
    def cropping(self, image):
        """Cropping the image to reduce computation"""
        radius = int(self.model.params[2])
        
        # Calculating the bounding box limits of the sun
        boundsX = [int(self.model.params[1] - radius), int(self.model.params[1] + radius)]
        boundsY = [int(self.model.params[0] - radius), int(self.model.params[0] + radius)]
        bounds = [boundsX[0], boundsX[1], boundsY[0], boundsY[1]]
        
        # Cropping the image
        image = image[boundsY[0]:boundsY[1],boundsX[0]:boundsX[1]]
        
        # Making all pixels outside of the radius white to remove the border of the sun
        for i in range(len(image)):
            for j in range(len(image)):
                if (i - radius)**2 + (j - radius)**2 > (0.95*radius)**2:
                    image[i][j] = 1
                    
        return image, radius, bounds
    
    def gaussMix(self, dataFrame, clusters):
        """Clustering the data to isolate the sunspots"""
        # Clustering
        gaussMix = sk.mixture.GaussianMixture(clusters, random_state=0, init_params="k-means++").fit(dataFrame)
        # Taking the Indices and creating a list of the unique values
        clusterIndices = gaussMix.predict(dataFrame)
            
        # Creating lists for storage
        clusteredX = []
        clusteredY = []
        
        # Checking if any clusters are adjacent to one another and combining if true
        touching = True
        while touching == True:
            adjacent = []
            for i in range(clusters):
                for j in range(i+1,clusters):
                    
                    if i == j: pass
                    
                    else:
                        cl1 = dataFrame[(clusterIndices==i)]
                        cl2 = dataFrame[(clusterIndices==j)]
                        
                        # Checking if the clusters have adjacent pixels
                        if isAdjacent(cl1, cl2):
                            adjacent.append([i,j])
            
            # Checking if the list of adjacent clusters is empty
            if len(adjacent) == 0:
                touching = False
                
            else:
                for pair in adjacent:
                    clusterIndices[clusterIndices == pair[0]] = pair[1]
                
        # Grouping the data in 2D arrays based on its cluster
        for cluster in np.unique(clusterIndices):
            if cluster in clusterIndices:
                clusteredX.append(dataFrame[(clusterIndices == cluster),0])
                clusteredY.append(dataFrame[(clusterIndices == cluster),1])
        
        uniqueIndices = np.arange(0,len(clusteredX))
        
        return clusteredX, clusteredY, uniqueIndices
            
    def plotting(self, ssCoords, bounds, filename):
        """Plotting the image and the sunspot coordinates"""
        # Showing the image with the fitting parameters
        plot = plotter(figsize=(16,12))
        plot.imshow(self.image)
        plot.aspect()
        
        # Plotting all of the sunspots over the image and labelling
        for i in range(len(ssCoords)):
            coords = ssCoords[i]
            plot.ax.annotate(f'{int(coords[2])}', xy=coords[0:2], textcoords='data')
            plot.scatter(coords[0:2], s=20, label = int(coords[2]))
        plot.limits(bounds[0],bounds[1],bounds[2],bounds[3])
        plot.invert("y")
        plot.save(f"data/images/{filename}.png")
        plot.scatter([self.rr,self.cc], c="r")
        plt.show()

# =============================================================================
# Data Analysis
# =============================================================================

class measurement():
    
    def initialise(self, sunspot, verbose = False, curtail = False):
        
        self.sunspot = sunspot
        
        # Reading the sunspot data file
        df = pd.read_csv(f"data/{sunspot}.csv", index_col=0)
        dates = df.index
        
        if curtail != False:
            dates = dates[curtail[0]:curtail[1]]
        
        # Creating lists for storage
        lats = []
        latsE = []
        lons = []
        lonsE = []
        dats = []
        
        # Calculating the heliographic latitudes and longitudes
        for date in dates:
            lat, lon, dat = measurement().analyse(date, sunspot, verbose)
            lats.append(lat[0])
            latsE.append(lat[1])
            lons.append(lon[0])
            lonsE.append(lon[1])
            dats.append(dat)
        dats = np.array(dats)
        
        # Plotting the data
        params, slopeErr = self.plotting(lats, lons, latsE, lonsE, dats)
        
        print(f"Rotational Period: {360/params[0]} +/- {(360/params[0]**2)*slopeErr}")
        latE = np.std(lats)/np.sqrt(len(lats))
        print(f"Average Latitude: {np.mean(lats).round(3)} +/- {latE}")
    
    def analyse(self, date, sunspot, verbose):
        # Reading the file containing sunspot positions
        self.sunspotDF = pd.read_csv(f"data/{sunspot}.csv", index_col=0).loc[date]
        
        # Pulling heliographic values
        self.B0 = sunpy.coordinates.sun.B0(time=date).radian
        self.L0 = sunpy.coordinates.sun.L0(time=date).radian
        self.P = sunpy.coordinates.sun.P(time=date).radian
        self.S = sunpy.coordinates.sun.angular_radius(t=date).radian
        
        # Checking which function to call for dataset type
            # Data taken through telescope
        if self.sunspotDF.type == "T":
            self.coordinateRotation(date, verbose)
            return self.latitude, self.longitude, self.date
            
            # Data taken from the SDO archive
        elif self.sunspotDF.type == "A":
            self.SDO(date)
            return self.latitude, self.longitude, self.date
        
        else:
            print("Invalid Entry")
        
    def SDO(self, date):
        
        # Pulling information from the datafile
        ssIndex = self.sunspotDF.sunspot
        
        # Setting the North Vector
        self.northMag = 1
        self.northMagErr = 0
        self.sunNorthV = np.array([0,1])
        
        ssCoordDF = pd.read_csv(f"data/{date}_ss.csv", index_col=0)
        data = pd.read_csv(f"data/{date}.csv")
        
        # Setting the centre coordinates of the sun's disc for translation
        translation = data.loc[0,["xPos", "yPos"]].to_numpy()
        
        # Translating the sunspot coordinates
        self.sunspotV = ssCoordDF.loc[ssIndex,["x","y"]].to_numpy()
        self.sunspotV = self.sunspotV - translation
        
        # Error in sunspot array
        self.sunspotVErr = np.array([2, 2])
        posE = np.linalg.norm(ssCoordDF.loc[ssIndex,["xE","yE"]])
        self.error = posE
        
        # Setting the sun's radius
        self.radius = data.radius[0]
        self.radiusErr = self.error
        
        # Pulling the julian date
        self.date = time2JulDate(date[0:4]+"-"+date[4:6]+"-"+date[6:8]+"T"+date[9:11]+":"+date[11:13]+":"+date[13:15])
        
        # Converting to heliographic coordinates
        self.heliographicConversion()
        
    def coordinateRotation(self, date, verbose):
        
        # Pulling information from the datafile
        ssIndex = self.sunspotDF.sunspot
        
        ssCoordDF = pd.read_csv(f"data/{date}_ss.csv", index_col=0)
        
        ssCoords = ssCoordDF.loc[ssIndex,["x","y"]].to_numpy()
        posE = np.linalg.norm(ssCoordDF.loc[ssIndex,["xE","yE"]])
        self.error = posE
        
        data = pd.read_csv(f"data/{date}.csv")
        imgIndex = np.where(data.time == date[0:4]+"-"+date[4:6]+"-"+date[6:8]+"T"+date[9:11]+":"+date[11:13]+":"+date[13:15])[0][0]
        
        self.radius = data.radius.loc[imgIndex]
        self.radiusErr = np.sqrt(np.sum(data.radius-self.radius) **2 / len(data.radius)) / np.sqrt(len(data.radius))
        
        # Pulling out the x,y coordinates of the suns disc
        xPos = data.xPos
        yPos = data.yPos
        
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
        motionInterceptErr = np.sqrt((motionSlope * posE)**2 + posE**2)
        
        if verbose == True:
            # Plotting the motion of the sun across the field of view
            plot = plotter()
            plot.defaultScatter([xPos, yPos], ["x Position", "y Position"])
            plot.plot([xPos, xPos*motionSlope+motionIntercept])

        # Calculating a vector pointing north, perpendicular to the apparent motion of the sun
        northSlope, northSlopeErr = (-1/motionSlope), ((1/motionSlope**2)*motionSlopeErr)
        northV = [50000, northSlope * 50000 + motionIntercept]
        northVErr = [posE, np.sqrt((50000*northSlopeErr)**2 + motionInterceptErr**2 + posE**2)]
        
        # Calculating the magnitude of the north vector and its error
        northMag = np.linalg.norm(northV)
        northMagErr1 = northV[0] * northVErr[1] / northMag
        northMagErr2 = northV[1] * northVErr[1] / northMag
        northMagErr = np.linalg.norm([northMagErr1, northMagErr2])
        
        # Setting attributes to be used for heliographic coordinate conversion
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
        sunspotVEx = (posE*np.cos(phi))**2 + (posE*np.sin(phi))**2 + ((sunspotV[0]*np.sin(phi) - sunspotV[1]*np.cos(phi))*psiErr)**2
        sunspotVEy = (posE*np.cos(phi))**2 + (posE*np.sin(phi))**2 + ((sunspotV[0]*np.cos(phi) - sunspotV[1]*np.sin(phi))*psiErr)**2
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
        
        # Calculating the angle between the solar north pole 
        # and the projection of the earth's north pole
        rho = np.sin(ssVectMag / self.radius) - (ssVectMag * self.S) / self.radius
        
        rhoErr1 = ((np.cos(ssVectMagErr / self.radius) - self.S) / self.radius)*ssVectMagErr
        rhoErr2 = ((self.S * ssVectMag - np.cos(ssVectMag / self.radius) * ssVectMag) / self.radius**2) * self.radiusErr
        rhoErr = np.linalg.norm([rhoErr1, rhoErr2])
        
        # Calculating the angle between the solar north pole 
        # and the sunspot vector
        chiTop = np.dot(sunspotV, sunNorthV)
        chiBottom = (ssVectMag * np.linalg.norm(sunNorthV))
        negativeChi = np.arccos(chiTop / chiBottom)
        
        # Correcting the angle such that it is always taken anticlockwise
        if sunspotV[0] >= 0:
            negativeChi = 2*np.pi - negativeChi
        
        chi = -negativeChi
        
        # Calculating the error in chi
        bottomErr = self.northMag * ssVectMag * np.sqrt(1 - (chiTop / chiBottom)**2)
        
        chiErrS = sunNorthV * sunspotVErr / bottomErr
        chiErrNy = (sunNorthV[1]*sunspotV[1])/(bottomErr)
        chiErrR = (chiTop*ssVectMagErr)/(bottomErr*ssVectMag)
        chiErrN = (chiTop*self.northMagErr)/(bottomErr*self.northMag)
        chiErr = np.sqrt(chiErrS[0]**2 + chiErrS[1]**2 + chiErrNy**2 + chiErrR**2 + chiErrN**2)
        
        # =====================================================================
        #         Calculating heliographic coordinates
        # =====================================================================
        
        # Finding Heliographic Latitude, B
        B1 = np.sin(self.B0)*np.cos(rho)
        B2 = np.cos(self.B0)*np.sin(rho)*np.cos(chi)
        B = np.arcsin(B1+B2)
        
        # Calculating the error in B
        BErrBottom = np.sqrt(1 - (np.cos(self.B0)*np.sin(rho)*np.cos(negativeChi) + np.sin(self.B0)*np.cos(rho))**2)
        BErrChi = (np.cos(self.B0)*np.sin(rho)*np.cos(negativeChi)*chiErr)/BErrBottom
        BErrRho = ((np.cos(self.B0)*np.cos(rho)*np.cos(negativeChi) - np.sin(self.B0)*np.sin(rho))*rhoErr)/BErrBottom
        BErr = np.sqrt(BErrChi**2 + BErrRho**2)
        
        # Finding Heliographic Latitude, L-L0
        inside = np.sin(rho) * np.sin(chi) * (1/np.cos(B))
        LL0 = np.rad2deg(np.arcsin(inside))
        
        # Calculating the error in L-L0
        LL0ErrBottom = np.sqrt(1 - inside**2)
        LL0ErrB = (inside * np.tan(B) * BErr) / (LL0ErrBottom)
        LL0ErrChi = (np.sin(rho) * np.cos(chi) * (1 / np.cos(B)) * chiErr) / (LL0ErrBottom)
        LL0ErrRho = (np.cos(rho) * np.sin(chi) * (1 / np.cos(B)) * rhoErr) / (LL0ErrBottom)
        LL0Err = np.rad2deg(np.sqrt(LL0ErrB**2 + LL0ErrChi**2 + LL0ErrRho**2))
        
        self.longitude = [LL0, LL0Err]
        self.latitude = [np.rad2deg(B), np.rad2deg(BErr)]

    def plotting(self, latitude, longitude, latE, lonE, dates):
        
        # Plotting the latitudes
        latplot = plotter(title=self.sunspot)
        latplot.defaultScatter([dates, latitude], ["Julian Date", "Latitude"])
        latplot.errorbar([dates, latitude], [None, latE])
        
        params, error = curve_fit(lin, dates, longitude, sigma = lonE)
        slopeErr = np.sqrt(np.diag(error))[0]
        
        # Plotting the longitudes
        lonplot = plotter(title=self.sunspot)
        # lonplot.errorbar([dates, longitude], [None, lonE])
        lonplot.plot([dates, dates*[params[0]] + params[1]])
        lonplot.defaultScatter([dates, longitude], ["Julian Date", "Longitude"])
        lonplot.ax.ticklabel_format(useOffset=False, style="plain")
        
        return params, slopeErr
        
measurement().initialise("ss_20250804-20250815")
# imageProcessing().sdoData("20250803-20250816")
# imageProcessing().imageSequence("2025-08-02-00")