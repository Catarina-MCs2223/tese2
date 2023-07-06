import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#metadata
metadata=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/Nova pasta/metadata.csv')
metadata.drop([0,5,29,37], axis=0, inplace=True)
metadata.to_csv('meta.csv', index=False)
meta=pd.read_csv('meta.csv')
#print(meta.columns.values.tolist())

#No_of_ACs é 18º coluna
#print(meta.iloc[:,18])
#print(meta.iloc[:,28])
#Soma de todos os aparelhos considerados de altos consumo energetico

meta['Sum_devices']=meta.iloc[:,[18,19,20,21,22,23,24,25,26,27,28]].sum(axis=1)
meta.drop(meta.columns[[18,19,20,21,22,23,24,25,26,27,28]], axis=1, inplace=True)
#print(meta)

#codificar a coluna da area (0 se for menor que 2858, 1 se for maior)
area=meta.iloc[:,1]
area = np.where(area >= 2858, 1, 0)
meta['Property_Area_sqft']=area

#codificar a coluna da ano (0 se for menor que 2005, 1 se for maior)
ano=meta.iloc[:,2]
ano = np.where(ano >= 2005, 1, 0)
meta['Building_Year']=ano

#apagar coluna 'altura do teto'
meta.drop(meta.columns[[3]], axis=1, inplace=True)

#apagar colunas a especificar divisões
meta.drop(meta.columns[[6,7,8,9]], axis=1, inplace=True)

#apagar a coluna do tipo de 'phase'
meta.drop(meta.columns[[6]], axis=1, inplace=True)


ins=meta.iloc[:,4]
ins = np.where(ins=='No', 1, 0)
meta['Ceiling_Insulation']=ins

'''pd.set_option('display.max_columns', None)
print(meta)'''

from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

km = TimeSeriesKMeans(n_clusters=2, metric="euclidean")
labels = km.fit_predict(meta)

print(km.labels_)
# Silhoutte Score
from sklearn.metrics import silhouette_score
score = silhouette_score(meta, km.labels_,metric='euclidean')
print('Silhouetter Score: %.3f' % score)

fancy_names_for_labels = [f"Cluster {label}" for label in labels]
end=pd.DataFrame(zip(houses,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Series").set_index("Series")
print(end)
