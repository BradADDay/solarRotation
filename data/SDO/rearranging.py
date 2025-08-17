
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join


files = [f for f in listdir("./") if isfile(join("./", f)) and ".txt" in f]
files.sort()
print(files)

for infile in files:
    f = open(infile, "r")
    file = f.read()
    f.close()
    file = file.replace("Timestamps: [", "")
    file = file.replace("']\n", "")
    file = file.replace("Sunspot Coords: ", "")
    file = file.replace("\n\nRadius of Disc: 395\n\nCentre: 513, 513", "")
    file = file.replace("'", "")
    file = file.split("\n")
    file[1] = file[1][1:-1]
    file[1] = file[1].split("], [")
    file[0] = file[0].split(", ")
    
    coords = file[1]
    for i in range(len(coords)):
        coords[i] = coords[i].split(", ")
    coords = np.array(coords)
    print(coords)
    
    df = pd.DataFrame(coords)
    df.insert(0, "date", file[0])
    df.insert(1, "imgIndex", None)
    df.insert(4, "type", "A")
    df.columns = ["", "imgIndex", "x", "y", "type"]
    df.set_index("", inplace=True)
    print(df)
    
    df.to_csv(f"{infile[:-4]}.csv")

