import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#dataset 'House 2'
h2=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House02.csv')
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
h2.to_csv('h2_day.csv', index=False)

h2_day=pd.read_csv('h2_day.csv')
'''print(h3_day.describe())
print(h3_day.head())'''
h2_day['DayOfWeek'] = pd.to_datetime(h2_day['Date']).dt.weekday

#weekend
saturday=h2_day.loc[h2_day['DayOfWeek']==4]
sunday=h2_day.loc[h2_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h2_day.loc[h2_day['DayOfWeek']==6]
week2=h2_day.loc[h2_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h2_day.iloc[30:122,:]
winter=h2_day.iloc[122:273,:]
print(len(summer))
#spring
spring1=h2_day.iloc[:30]
spring2=h2_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)
print(len(spring))
print(len(winter))

print("Summer h2:" + str(summer.mean()) )
print("Winter h2:" + str(winter.mean()))
print("Spring h2:" + str(spring.mean()))
print("Weekend h2:" + str(weekends.mean()) )
print("Week Days h2:" + str(week_days.mean()))
print("Spring h2:" + str(spring.mean()))

#grafico de linhas para analisar picos ao longo do ano
plt.plot(summer['Date'],summer['Usage_kW'], color='red')
plt.plot(winter['Date'],winter['Usage_kW'], color='blue')
plt.plot(spring['Date'],spring['Usage_kW'], color='orange')
#plt.plot(h1['Date'],h1['sum'], color='green')
plt.title('daily usage - house_1', fontsize=10)
plt.xlabel('day', fontsize=12)
plt.ylabel('Usage (kW)', fontsize=12)
#plt.grid(True)'''
plt.show()

#House 04

h8=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House08.csv')
h8['Date'] = pd.to_datetime(h8['Date_Time']).dt.date
h8['Time'] = pd.to_datetime(h8['Date_Time']).dt.time
h8 = h8.drop("Date_Time", axis='columns')
data=h8['Date'].unique()
h8=h8.groupby(['Date']).sum()
h8.insert(loc=0,column='Date',value=data)
h8.to_csv('h8_day.csv', index=False)

h8_day=pd.read_csv('h8_day.csv')
h8_day['DayOfWeek'] = pd.to_datetime(h8_day['Date']).dt.weekday

#weekend
saturday=h8_day.loc[h8_day['DayOfWeek']==4]
sunday=h8_day.loc[h8_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h8_day.loc[h8_day['DayOfWeek']==6]
week2=h8_day.loc[h8_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h8_day.iloc[30:122,:]
winter=h8_day.iloc[122:273,:]

#spring
spring1=h8_day.iloc[:30]
spring2=h8_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h8_day.describe())
print("Summer h8:" + str(summer.mean()) )
print("Winter h8:" + str(winter.mean()))
print("Spring h8:" + str(spring.mean()))
print("Weekend h8:" + str(weekends.mean()) )
print("Week Days h8:" + str(week_days.mean()))

#House 15
h10=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House10.csv')
h10['Date'] = pd.to_datetime(h10['Date_Time']).dt.date
h10['Time'] = pd.to_datetime(h10['Date_Time']).dt.time
h10 = h10.drop("Date_Time", axis='columns')
data=h10['Date'].unique()
h10=h10.groupby(['Date']).sum()
h10.insert(loc=0,column='Date',value=data)
h10.to_csv('h10_day.csv', index=False)

h10_day=pd.read_csv('h10_day.csv')
h10_day['DayOfWeek'] = pd.to_datetime(h10_day['Date']).dt.weekday

#weekend
saturday=h10_day.loc[h10_day['DayOfWeek']==4]
sunday=h10_day.loc[h10_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h10_day.loc[h10_day['DayOfWeek']==6]
week2=h10_day.loc[h10_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h10_day.iloc[30:122,:]
winter=h10_day.iloc[122:273,:]

#spring
spring1=h10_day.iloc[:30]
spring2=h10_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h10_day.describe())
print("Summer h10:" + str(summer.mean()) )
print("Winter h10:" + str(winter.mean()))
print("Spring h10:" + str(spring.mean()))
print("Weekend h10:" + str(weekends.mean()) )
print("Week Days h10:" + str(week_days.mean()))

#House 16
h14=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House14.csv')
h14['Date'] = pd.to_datetime(h14['Date_Time']).dt.date
h14['Time'] = pd.to_datetime(h14['Date_Time']).dt.time
h14 = h14.drop("Date_Time", axis='columns')
data=h14['Date'].unique()
h14=h14.groupby(['Date']).sum()
h14.insert(loc=0,column='Date',value=data)
h14.to_csv('h14_day.csv', index=False)

h14_day=pd.read_csv('h14_day.csv')
h14_day['DayOfWeek'] = pd.to_datetime(h14_day['Date']).dt.weekday

#weekend
saturday=h14_day.loc[h14_day['DayOfWeek']==4]
sunday=h14_day.loc[h14_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h14_day.loc[h14_day['DayOfWeek']==6]
week2=h14_day.loc[h14_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h14_day.iloc[30:122,:]
winter=h14_day.iloc[122:273,:]

#spring
spring1=h14_day.iloc[:30]
spring2=h14_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h14_day.describe())
print("Summer h14:" + str(summer.mean()) )
print("Winter h14:" + str(winter.mean()))
print("Spring h14:" + str(spring.mean()))
print("Weekend h14:" + str(weekends.mean()) )
print("Week Days h14:" + str(week_days.mean()))

#House 19

h17=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House17.csv')
h17['Date'] = pd.to_datetime(h17['Date_Time']).dt.date
h17['Time'] = pd.to_datetime(h17['Date_Time']).dt.time
h17 = h17.drop("Date_Time", axis='columns')
data=h17['Date'].unique()
h17=h17.groupby(['Date']).sum()
h17.insert(loc=0,column='Date',value=data)
h17.to_csv('h17_day.csv', index=False)

h17_day=pd.read_csv('h17_day.csv')
h17_day['DayOfWeek'] = pd.to_datetime(h17_day['Date']).dt.weekday

#weekend
saturday=h17_day.loc[h17_day['DayOfWeek']==4]
sunday=h17_day.loc[h17_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h17_day.loc[h17_day['DayOfWeek']==6]
week2=h17_day.loc[h17_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h17_day.iloc[30:122,:]
winter=h17_day.iloc[122:273,:]

#spring
spring1=h17_day.iloc[:30]
spring2=h17_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h17_day.describe())
print("Summer h17:" + str(summer.mean()) )
print("Winter h17:" + str(winter.mean()))
print("Spring h17:" + str(spring.mean()))
print("Weekend h17:" + str(weekends.mean()) )
print("Week Days h17:" + str(week_days.mean()))

#House 28

h22=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House22.csv')
h22['Date'] = pd.to_datetime(h22['Date_Time']).dt.date
h22['Time'] = pd.to_datetime(h22['Date_Time']).dt.time
h22 = h22.drop("Date_Time", axis='columns')
data=h22['Date'].unique()
h22=h22.groupby(['Date']).sum()
h22.insert(loc=0,column='Date',value=data)
h22.to_csv('h22_day.csv', index=False)

h22_day=pd.read_csv('h22_day.csv')
h22_day['DayOfWeek'] = pd.to_datetime(h22_day['Date']).dt.weekday

#weekend
saturday=h22_day.loc[h22_day['DayOfWeek']==4]
sunday=h22_day.loc[h22_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h22_day.loc[h22_day['DayOfWeek']==6]
week2=h22_day.loc[h22_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h22_day.iloc[30:122,:]
winter=h22_day.iloc[122:273,:]

#spring
spring1=h22_day.iloc[:30]
spring2=h22_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h22_day.describe())
print("Summer h22:" + str(summer.mean()) )
print("Winter h22:" + str(winter.mean()))
print("Spring h22:" + str(spring.mean()))
print("Weekend h22:" + str(weekends.mean()) )
print("Week Days h22:" + str(week_days.mean()))
#House 34

h24=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House24.csv')
h24['Date'] = pd.to_datetime(h24['Date_Time']).dt.date
h24['Time'] = pd.to_datetime(h24['Date_Time']).dt.time
h24 = h24.drop("Date_Time", axis='columns')
data=h24['Date'].unique()
h24=h24.groupby(['Date']).sum()
h24.insert(loc=0,column='Date',value=data)
h24.to_csv('h24_day.csv', index=False)

h24_day=pd.read_csv('h24_day.csv')
h24_day['DayOfWeek'] = pd.to_datetime(h24_day['Date']).dt.weekday

#weekend
saturday=h24_day.loc[h24_day['DayOfWeek']==4]
sunday=h24_day.loc[h24_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h24_day.loc[h24_day['DayOfWeek']==6]
week2=h24_day.loc[h24_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h24_day.iloc[30:122,:]
winter=h24_day.iloc[122:273,:]

#spring
spring1=h24_day.iloc[:30]
spring2=h24_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h24_day.describe())
print("Summer h24:" + str(summer.mean()) )
print("Winter h24:" + str(winter.mean()))
print("Spring h24:" + str(spring.mean()))
print("Weekend h24:" + str(weekends.mean()) )
print("Week Days h24:" + str(week_days.mean()))
#House 40

h41=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House41.csv')
h41['Date'] = pd.to_datetime(h41['Date_Time']).dt.date
h41['Time'] = pd.to_datetime(h41['Date_Time']).dt.time
h41 = h41.drop("Date_Time", axis='columns')
data=h41['Date'].unique()
h41=h41.groupby(['Date']).sum()
h41.insert(loc=0,column='Date',value=data)
h41.to_csv('h41_day.csv', index=False)

h41_day=pd.read_csv('h41_day.csv')
h41_day['DayOfWeek'] = pd.to_datetime(h41_day['Date']).dt.weekday

#weekend
saturday=h41_day.loc[h41_day['DayOfWeek']==4]
sunday=h41_day.loc[h41_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h41_day.loc[h41_day['DayOfWeek']==6]
week2=h41_day.loc[h41_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h41_day.iloc[30:122,:]
winter=h41_day.iloc[122:273,:]

#spring
spring1=h41_day.iloc[:30]
spring2=h41_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h41_day.describe())
print("Summer h41:" + str(summer.mean()) )
print("Winter h41:" + str(winter.mean()))
print("Spring h41:" + str(spring.mean()))
print("Weekend h41:" + str(weekends.mean()) )
print("Week Days h41:" + str(week_days.mean()))

#House 42

h42=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House42.csv')
h42['Date'] = pd.to_datetime(h42['Date_Time']).dt.date
h42['Time'] = pd.to_datetime(h42['Date_Time']).dt.time
h42 = h42.drop("Date_Time", axis='columns')
data=h42['Date'].unique()
h42=h42.groupby(['Date']).sum()
h42.insert(loc=0,column='Date',value=data)
h42.to_csv('h42_day.csv', index=False)

h42_day=pd.read_csv('h42_day.csv')
h42_day['DayOfWeek'] = pd.to_datetime(h42_day['Date']).dt.weekday

#weekend
saturday=h42_day.loc[h42_day['DayOfWeek']==4]
sunday=h42_day.loc[h42_day['DayOfWeek']==5]
weekends=[saturday, sunday]
weekends=pd.concat(weekends)
weekends=weekends.sort_values(by=['Date'])

#week day
week1=h42_day.loc[h42_day['DayOfWeek']==6]
week2=h42_day.loc[h42_day['DayOfWeek']<4]
week=[week1, week2]
week=pd.concat(week)
week_days=week.sort_values(by=['Date'])

summer=h42_day.iloc[30:122,:]
winter=h42_day.iloc[122:273,:]

#spring
spring1=h42_day.iloc[:30]
spring2=h42_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)

print(h42_day.describe())
print("Summer h42:" + str(summer.mean()) )
print("Winter h42:" + str(winter.mean()))
print("Spring h42:" + str(spring.mean()))
print("Weekend h42:" + str(weekends.mean()) )
print("Week Days h42:" + str(week_days.mean()))