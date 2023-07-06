import pandas as pd

#datset 'House 1'
house1=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/House1.csv')

#colunas
#column_names = list(house1.columns)
#print(column_names)

#describe
#print(house1.describe)

#separa a coluna "datatime" em 2 colunas separadas (dia/ horas)

house1['Date'] = pd.to_datetime(house1['Date_Time']).dt.date
house1['Time'] = pd.to_datetime(house1['Date_Time']).dt.time
house1 = house1.drop("Date_Time", axis='columns')


#ver uma linha específica
day1=house1.iloc[:1440]

day1.to_csv('day1.csv', index=False)
