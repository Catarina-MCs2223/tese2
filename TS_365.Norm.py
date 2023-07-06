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

#House02
#print(myHouses[1])
for i in range(len(myHouses)):
 myHouses[i]=myHouses[i].values/myHouses[i].values.sum()


'''fig, axs = plt.subplots(6,6,figsize=(30,25))
fig.suptitle('Series')
for i in range(6):
    for j in range(6):
        if i*6+j+1>len(myHouses):
            continue
        axs[i, j].plot(myHouses[i*6+j].values)
        axs[i, j].set_title(namesofMyHouses[i*6+j])
plt.show()
'''
series_lengths = {len(series) for series in myHouses}
print(series_lengths)

'''for i in range(len(myHouses)):
    myHouses[i]=myHouses[i].values/myHouses[i].values.sum()'''

#Time Series K-Means
cluster_count = math.ceil(math.sqrt(len(myHouses)))
km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
labels = km.fit_predict(myHouses)
myHouses=np.reshape(myHouses,(38,365))

# Silhoutte Score
from sklearn.metrics import silhouette_score
score = silhouette_score(myHouses, km.labels_,metric='euclidean')
print('Silhouetter Score: %.3f' % score)


#Plot
plot_count = math.ceil(math.sqrt(3)) #2

som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(myHouses)))) #3

'''fig, axs = plt.subplots(2, 2, figsize=(25, 25))
fig.suptitle('Clusters')
row_i = 0
column_j = 0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
        if (labels[i] == label):
            axs[row_i, column_j].plot(myHouses[i], c="gray", alpha=0.4)
            cluster.append(myHouses[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
    axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j))
    column_j += 1
    if column_j % plot_count == 0:
        row_i += 1
        column_j = 0'''


# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
        if (labels[i] == label):
            cluster.append(myHouses[i])
    if len(cluster) > 0:
        plt.plot(np.average(np.vstack(cluster), axis=0), c="red")

plt.show()
print(len(cluster))

fancy_names_for_labels = [f"Cluster {label}" for label in labels]
end=pd.DataFrame(zip(namesofMyHouses,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")
print(end)
