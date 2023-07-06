import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

#datset 'House 1'
house1=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/House01.csv')

#h1=h1.rename(columns={})

#shape
#print(house1.shape)

#dtypes
#print(house1.dtypes)

#describe
#print(house1.describe)

#separa a coluna "datatime" em 2 colunas separadas (dia/ horas)

house1['Date'] = pd.to_datetime(house1['Date_Time']).dt.date
house1['Time'] = pd.to_datetime(house1['Date_Time']).dt.time
house1 = house1.drop("Date_Time", axis='columns')
data=house1['Date'].unique()

#soma todos os valores por dia

h1=house1.groupby(['Date']).sum()


#colunas
'''column_names = list(h1.columns)
print(column_names)'''

#Rename columns
h1=h1.rename(columns={'Usage_kW':'Usage', 'AC_DR_kW':'AC_DR', 'UPS_kW':'UPS', 'LR_kW':'LR', 'Kitchen_kW':'Kitc', 'AC_DNR_kW':'AC_DNR', 'AC_BR_kW':'AC_BR'})


#adicionar uma coluna com as dias
h1.insert(loc=0,column='Date',value=data)
#print(h1.describe(include='all'))
'''#soma de todas as colunas menos 'usage'
values=h1.iloc[:,-6:]
sum= values.sum( axis = 1)

h1.insert(loc=8,column='sum',value=sum)
print(h1)
#h1.to_csv('h1_day.csv', index=False)

#grafico de linhas para analisar picos ao longo do ano
plt.plot(h1['Date'],h1['Usage'], color='red')
plt.plot(h1['Date'],h1['sum'], color='green')
plt.title('daily usage - house_1', fontsize=10)
plt.xlabel('day', fontsize=12)
plt.ylabel('Usage (kW)', fontsize=12)
#plt.grid(True)'''

'''#Correlações entre dados
#encoding data
objList = h1.select_dtypes(include = "object").columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feat in objList:
    h1[feat] = le.fit_transform(h1[feat].astype(str))

correlation=h1.corr()
sns.heatmap(correlation, cmap='GnBu',  linewidth=0.5)'''

#Pairplot
#sns.pairplot(h1, height=1)

#Box Plot
#sns.boxplot(data=h1, width=0.5)

#Histograma

#sns.barplot(h1)
#plt.show()