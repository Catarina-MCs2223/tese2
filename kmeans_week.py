import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''#dataset 'House 2'
h2=pd.read_csv('C:/Users/Catarina Lima/PycharmProjects/teste_bd/h42_day.csv')
h2 = h2.loc[:,["Date","Usage_kW"]]
h2['DayOfWeek'] = pd.to_datetime(h2['Date']).dt.weekday


print("monday h:" + str((h2.loc[h2['DayOfWeek']==0]).mean()) )
print("tuesday h:" + str((h2.loc[h2['DayOfWeek']==1]).mean()))
print("wednesday h:" + str((h2.loc[h2['DayOfWeek']==2]).mean()))
print("thursday h:" + str((h2.loc[h2['DayOfWeek']==3]).mean()))
print("friday h:" + str((h2.loc[h2['DayOfWeek']==4]).mean()))
print("saturday h:" + str((h2.loc[h2['DayOfWeek']==5]).mean()))
print("sunday h:" + str((h2.loc[h2['DayOfWeek']==6]).mean()))'''

week=pd.read_excel('C:/Users/Catarina Lima/Desktop/PRECON dataset/km_week.xlsx')
houses=week.iloc[:,0]

week.drop(week.columns[[2,3,4,5,6,7]], axis=1, inplace=True)
print(week.describe())

km = TimeSeriesKMeans(n_clusters=2, metric="euclidean")
labels = km.fit_predict(week)

print(km.labels_)
# Silhoutte Score
from sklearn.metrics import silhouette_score
score = silhouette_score(week, km.labels_,metric='euclidean')
print('Silhouetter Score: %.3f' % score)

fancy_names_for_labels = [f"Cluster {label}" for label in labels]
end=pd.DataFrame(zip(houses,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Series").set_index("Series")
print(end)



##################################################################################################################################
'''#Graph of silhouette score
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np


range_n_clusters = [2, 3, 4]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(week) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(week)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(week, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(week, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


plt.show()

###################################################################################################################
#Elbow graph

inertias = []

for i in range(1,6):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(week)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,6), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()'''