import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

h2=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/House42.csv')

h2['Date'] = pd.to_datetime(h2['Date_Time']).dt.date
h2['Time'] = pd.to_datetime(h2['Date_Time']).dt.time
h2 = h2.drop("Date_Time", axis='columns')
data=h2['Date'].unique()

#soma todos os valores por dia
h2=h2.groupby(['Date']).sum()

#colunas
column_names = list(h2.columns)
print(column_names)

#adicionar uma coluna com as dias
h2.insert(loc=0,column='Date',value=data)

h2.to_csv('h2_day.csv', index=False)


import matplotlib.pyplot as plt
print(h2.iloc[182,:])


data=h2['Date']
plt.plot(data,h2['Usage_kW'], linewidth=1.5)
plt.xticks(rotation=30, ha='right')
plt.axvline(x=data[153],color = 'red', linestyle='--')
plt.axvline(x=data[302],color = 'red', linestyle='--')
plt.axvline(x=data[30],color = 'red', linestyle='--')
plt.text(data[210], 1400, 'Estação Fria', fontweight='semibold', fontdict=None)
plt.text(data[65], 1400, 'Estação Quente', fontweight='semibold',fontdict=None)
plt.text(data[310], 1400, 'Estação Muito Quente', fontweight='semibold',fontdict=None)
plt.xlabel("Data")
plt.ylabel("Usage_kW")
plt.title("Consumo energético anual- House 42",fontweight='bold', fontsize='large')
plt.show()


