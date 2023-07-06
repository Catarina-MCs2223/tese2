import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#metadata
metadata=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/Nova pasta/metadata.csv')
print(metadata)
low13=metadata.iloc[13,:]
low16=metadata.iloc[16,:]
low21=metadata.iloc[21,:]
low23=metadata.iloc[23,:]
low40=metadata.iloc[40,:]
low41=metadata.iloc[41,:]
print(low13)
print(low16)
print(low21)
print(low23)
print(low40)
print(low41)




#dataset 'House 2'
h2=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/House02.csv')

#separa a coluna "datatime" em 2 colunas separadas (dia/ horas)

h2['Date'] = pd.to_datetime(h2['Date_Time']).dt.date
h2['Time'] = pd.to_datetime(h2['Date_Time']).dt.time
h2 = h2.drop("Date_Time", axis='columns')
data=h2['Date'].unique()

#soma todos os valores por dia
h2=h2.groupby(['Date']).sum()

#colunas
column_names = list(h2.columns)
print(column_names)

#Rename columns
h2=h2.rename(columns={'Usage_kW':'Usage', 'AC_DR_kW':'AC_DR', 'UPS_kW':'UPS', 'LR_kW':'LR', 'Kitchen_kW':'Kitc', 'AC_DNR_kW':'AC_DNR', 'AC_BR_kW':'AC_BR'})

#adicionar uma coluna com as dias
h2.insert(loc=0,column='Date',value=data)

h2.to_csv('h2_day.csv', index=False)

h12_day=pd.read_csv('C:/Users/Catarina Lima/PycharmProjects/teste_bd/venv/share/Normalizaçao/h12_day.csv')
#print(h2_day.head())

sum=h2_day.loc[:,:].sum()
print(sum[1]) #soma da coluna Usage

h2_day['Usage_norm']=h2_day.iloc[:,1]/h2_day.iloc[:,1].sum()

#h2_day['Usage']=h2_day.iloc[:,1]/h2_day.iloc[:,1].sum()

'''
#normalização
for i in range(1,7):
  h2_day[h2_day.columns[i]]=h2_day.iloc[:,i]/h2_day.iloc[:,i].sum()

pd.set_option('display.max_columns', None)
print(h2_day.describe(include='all'))'''

#print(h2_day)


#grafico de linhas para analisar picos ao longo do ano
plt.plot(h2_day['Date'],h2_day['Usage_norm'], color='red')
#plt.plot(h1['Date'],h1['sum'], color='green')
plt.title('daily usage - house_1', fontsize=10)
plt.xlabel('day', fontsize=12)
plt.ylabel('Usage (kW)', fontsize=12)
#plt.grid(True)'''
plt.show()


