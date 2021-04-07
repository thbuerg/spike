# Preprocessing.
import pandas as pd
import numpy as np
# import pytorch.Data.Dataset
path = '/Users/buergelt/Projects/timESD/timESD/source'

# import grid load data
df1 = pd.read_csv(f"{path}/data/17-2-1 HS 2020-Q1.csv", sep=';', decimal=',', header=7)
df1.dropna(how='all', inplace=True)
df2 = pd.read_csv(f"{path}/data/17-2-1 HS 2020-Q2.csv", sep=';', decimal=',', header=7)
df2.dropna(how='all', inplace=True)
df3 = pd.read_csv(f"{path}/data/17-2-1 HS 2020-Q3.csv", sep=';', decimal=',', header=7)
df3.dropna(how='all', inplace=True)
df4 = pd.read_csv(f"{path}/data/17-2-1 HS 2020-Q4.csv", sep=';', decimal=',', header=7)
df4.dropna(how='all', inplace=True)
df = pd.concat([df1, df2, df3, df4])

df['datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['von'])
df.set_index('datetime', inplace=True)
df.drop(['Datum', 'von', 'bis'], axis=1, inplace=True)
newindex = pd.DatetimeIndex(freq='15T', data = pd.date_range(start=df.index[0], periods=35136, freq='15T'))
df.set_index(newindex, inplace=True)



# import weather data
temp = pd.read_csv(f"{path}/data/produkt_zehn_min_tu_20191005_20210406_01270.txt", sep=';')
temp['datetime']=pd.to_datetime(temp.MESS_DATUM, format='%Y%m%d%H%M')
temp.set_index('datetime', inplace=True)
temp.drop(['STATIONS_ID', 'MESS_DATUM', '  QN', 'PP_10', 'TM5_10', 'RF_10', 'TD_10', 'eor'], axis=1, inplace=True)
temp = temp.resample("15T").mean()
temp = temp.loc['2020',:]


wind = pd.read_csv(f"{path}/data/produkt_zehn_min_ff_20191005_20210406_01270.txt", sep=';')
wind['datetime']=pd.to_datetime(wind.MESS_DATUM, format='%Y%m%d%H%M')
wind.set_index('datetime', inplace=True)
wind.drop(['STATIONS_ID', 'MESS_DATUM', '  QN', 'DD_10', 'eor'], axis=1, inplace=True)
wind = wind.resample("15T").mean()
wind = wind.loc['2020',:]

sun = pd.read_csv(f"{path}/data/produkt_zehn_min_sd_20191005_20210406_01270.txt", sep=';')
sun['datetime']=pd.to_datetime(sun.MESS_DATUM, format='%Y%m%d%H%M')
sun.set_index('datetime', inplace=True)
sun.drop(['STATIONS_ID', 'MESS_DATUM', '  QN', 'DS_10', 'SD_10', 'LS_10', 'eor'], axis=1, inplace=True)
sun = sun.resample("15T").mean()
sun = sun.loc['2020',:]
sun.loc[sun.GS_10==-999] = np.nan
sun.iloc[0,0]=0
sun = sun.fillna(method='ffill')

data = pd.concat([df, temp, sun, wind], axis=1)

# set daily max of grid load to 1, everything else to 0
dailymax = data.groupby(pd.Grouper(freq='D'))['Wert'].idxmax()
data.Wert=0
data.loc[dailymax,'Wert']=1
data.index.name = 'date_time'
data.to_csv(f'{path}/data/data.csv')







