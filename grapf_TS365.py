import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Algorithms
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

directory = '/Users/Catarina Lima/Desktop/PRECON dataset/365/'

myHouses = []
namesofMyHouses = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(directory+filename)
        df = df.loc[:,["Date_Time","Usage_kW"]]
        # While we are at it I just filtered the columns that we will be working on
        #df.set_index("Date_Time",inplace=True)
        df['Date'] = pd.to_datetime(df['Date_Time']).dt.date
        df['Time'] = pd.to_datetime(df['Date_Time']).dt.time
        df = df.drop("Date_Time", axis='columns')
        data = df['Date'].unique()
        #df = df['Date'].unique()
        df = df.groupby(['Date']).sum()
        df.insert(loc=0, column='Date', value=data)
        df.set_index("Date", inplace=True)
        # ,set the date columns as index
        df.sort_index(inplace=True)
        # and lastly, ordered the data according to our date index
        myHouses.append(df)
        namesofMyHouses.append(filename[:-4])

for i in range(len(myHouses)):
    plt.plot(np.average(myHouses[i], axis=0), c="red")
    plt.show()