import pandas as pd
from sklearn import preprocessing
from pandas import read_csv
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#dataset 'House 2'
h=pd.read_csv('C:/Users/Catarina Lima/Desktop/PRECON dataset/365/House07.csv')
h['Date'] = pd.to_datetime(h['Date_Time']).dt.date
h['Time'] = pd.to_datetime(h['Date_Time']).dt.time
h = h.drop("Date_Time", axis='columns')
data=h['Date'].unique()
h=h.groupby(['Date']).sum()
h.insert(loc=0,column='Date',value=data)
h.to_csv('h39_day.csv', index=False)
h_day=pd.read_csv('h7_day.csv')
h_day['DayOfWeek'] = pd.to_datetime(h_day['Date']).dt.weekday

summer=h_day.iloc[30:122,:]
winter=h_day.iloc[122:273,:]

#spring
spring1=h_day.iloc[:30]
spring2=h_day.iloc[273:]
data=[spring1, spring2]
spring=pd.concat(data)


print("Summer h:" + str(summer.mean()) )
print("Winter h:" + str(winter.mean()))
print("Spring h:" + str(spring.mean()))




'''#weekend
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
print("Weekend h2:" + str(weekends.mean()) )
print("Week Days h2:" + str(week_days.mean()))'''