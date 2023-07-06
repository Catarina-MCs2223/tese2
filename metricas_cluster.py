import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import openpyxl
import seaborn as sns
import matplotlib.cm as cm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, silhouette_samples

km_dias = pd.read_excel('C:/Users/Catarina Lima/Desktop/PRECON dataset/kmeans_dias2.xlsx')
km_est = pd.read_excel('C:/Users/Catarina Lima/Desktop/PRECON dataset/kmeans_estaçao.xlsx')

########### Silhouette Score ################################

'''range_n_clusters = [2, 3, 4]

House = km_dias['House']
mean = km_dias['Mean']
CV = km_dias['CV']

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(km_dias) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(km_dias)

    silhouette_avg = silhouette_score(km_dias, cluster_labels)
    print("For n_clusters =",n_clusters,"The average silhouette_score is :",silhouette_avg,)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(km_dias, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color,edgecolor=color,
            alpha=0.7,)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(House, mean, marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=100, edgecolor="k", )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=25, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %2d"
        % n_clusters,
        fontsize=10,
        fontweight="bold",
    )
plt.show()'''

########### Elbow method ################################
inertias = []

'''for i in range(1,6):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(km_dias)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,6), inertias, marker='o')
plt.title('Elbow method km_dias')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()'''

for i in range(1,6):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(km_est)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,6), inertias, marker='o')
plt.title('Elbow method km_est')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
########### Hierarchical clustering ################################
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

data=list(zip(km_dias['House'], km_dias['Mean']))
print(data)

#Ward (variance)
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.axhline(y=2510, color='r', linestyle='--')
plt.axhline(y=5090, color='r', linestyle='--')
plt.axhline(y=1510, color='r', linestyle='--')
plt.axvline(x=3,color='b',ls='-')
plt.show()

#Complete (max dist.)
linkage_data = linkage(data, method='complete', metric='euclidean')
dendrogram(linkage_data)
plt.axhline(y=1985, color='r', linestyle='--')
plt.axhline(y=3032, color='r', linestyle='--')
plt.axhline(y=998, color='r', linestyle='--')
plt.axvline(x=3,color='b',ls='-')
plt.show()