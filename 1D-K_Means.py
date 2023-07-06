import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import openpyxl
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Algorithms
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score


#Cluster do dataset em dias, grafico 2D (Média e Coeficiente de Variação)
km_dias = pd.read_excel('C:/Users/Catarina Lima/Desktop/PRECON dataset/kmeans_dias2.xlsx')

#km_dias.plot.scatter( x='Mean',y='CV',color='DarkBlue')
#plt.show()

#sns.scatterplot(data=km_dias, x='Mean', y='CV')

kmeans=KMeans(n_clusters=2, random_state=0, n_init='auto')
kmeans.fit(km_dias)

labels_dias= kmeans.labels_
score=silhouette_score(km_dias,labels_dias, metric='euclidean')
print('Silhouetter Score: %.3f' % score)

fancy_names_for_labels = [f"Cluster {label}" for label in labels_dias]
end=pd.DataFrame(zip(range(1,39),fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Series").set_index("Series")
print(end)

#print(kmeans.labels_)
#print(kmeans.cluster_centers_)

########################################################################################################################

#Cluster dataset com os valores medios de energia gasta por estação, grafico 3D (Summer, Winter e Spring)
km_est = pd.read_excel('C:/Users/Catarina Lima/Desktop/PRECON dataset/kmeans_estaçao.xlsx')
#ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

#print(km_est.describe())

#3D-Graph
'''summer=km_est['Summer']
winter=km_est['Winter']
spring=km_est['Spring']
ax = plt.axes(projection='3d')
ax.scatter3D(summer, winter, spring,c=spring, cmap='Greens')
ax.set_xlabel('Summer')
ax.set_ylabel('Winter')
ax.set_zlabel('Spring');
#plt.show()'''

kmeans=KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(km_est)

labels_est= kmeans.labels_
score=silhouette_score(km_est,labels_est, metric='euclidean')
print('Silhouetter Score: %.3f' % score)

fancy_names_for_labels = [f"Cluster {label}" for label in labels_est]
end=pd.DataFrame(zip(range(1,39),fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Series").set_index("Series")
print(end)

########################################################################################################################

#Verificar as "Houses" onde o cluster deu igual por media anual vs. média por estação
print(labels_dias==labels_est)

########################################################################################################################

#Cluster do dataset em dias, grafico 2D (Média e Coeficiente de Variação)
km_dias = pd.read_excel('C:/Users/Catarina Lima/Desktop/PRECON dataset/kmeans_dias2.xlsx')
km_mean=km_dias.drop(['CV'],axis='columns')

kmeans=KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(km_mean)

labels_mean= kmeans.labels_
score=silhouette_score(km_dias,labels_mean, metric='euclidean')
print('Silhouetter Score: %.3f' % score)

fancy_names_for_labels = [f"Cluster {label}" for label in labels_mean]
end=pd.DataFrame(zip(range(1,39),fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Series").set_index("Series")
print(end)
