import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#dataset 'House 2'
h2=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House15.csv')
h2 = h2.loc[:,["Date_Time","Usage_kW"]]

#separa a coluna "datatime" em 2 colunas separadas (dia/ horas)

h2['Date'] = pd.to_datetime(h2['Date_Time']).dt.date
h2['Time'] = pd.to_datetime(h2['Date_Time']).dt.time
h2 = h2.drop("Date_Time", axis='columns')
data=h2['Date'].unique()

#soma todos os valores por dia
h2=h2.groupby(['Date']).sum()

#Rename columns
#h2=h2.rename(columns={'Usage_kW':'Usage', 'AC_DR_kW':'AC_DR', 'UPS_kW':'UPS', 'LR_kW':'LR', 'Kitchen_kW':'Kitc', 'AC_DNR_kW':'AC_DNR', 'AC_BR_kW':'AC_BR'})

#adicionar uma coluna com as dias
h2.insert(loc=0,column='Date',value=data)
h2.to_csv('h15_day.csv', index=False)

h2_day=pd.read_csv('h15_day.csv')