#Written by Bradley Day (jq23394)
#importing libraries
from os import listdir
from os.path import isfile, join

#stating the directory and reading all files in that directory
path = "2025-08-02"
files = [f for f in listdir(f"imageSets/{path}") if isfile(join(f"imageSets/{path}", f))]
files.sort()
f = "image: \nssCoord: \ntime,radius,xPos,yPos\n"

#taking the date and time from the filename
for file in files:
    date = file[:8]
    time = file[9:15]
    f += date[0:4]+"-"+date[4:6]+"-"+date[6:8]+"T"+time[0:2]+":"+time[2:4]+":"+time[4:6]+",,,\n"
    
#writing the dates and times to a text file
textFile = open(f"data/{path}.csv", "w")
textFile.write(str(f))
textFile.close()
