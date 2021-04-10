# Preprocessing.
import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests, zipfile, io

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='./config/', config_name="spike.yml")
def main(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    os.makedirs(FLAGS.experiment.file_path, exist_ok=True)

    # https://www.thueringer-energienetze.com/Content/Documents/Ueber_uns/p_17_2-1_MS_2020.zip
    # https://www.thueringer-energienetze.com/Content/Documents/Ueber_uns/p_17_2-1_HSU_2020.zip
    load_url = 'https://www.thueringer-energienetze.com/Content/Documents/Ueber_uns/p_17_2-1_HS_2020.zip'
    r = requests.get(load_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df_list = []
    for f in z.namelist():
        df_tmp = pd.read_csv(z.open(f), sep=';', decimal=',', header=7, encoding='latin1')
        df_tmp.dropna(how='all', inplace=True)
        df_list.append(df_tmp)
    df = pd.concat(df_list)
    df['datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['von'])
    df.set_index('datetime', inplace=True)
    df.drop(['Datum', 'von', 'bis'], axis=1, inplace=True)
    newindex = pd.DatetimeIndex(freq='15T', data = pd.date_range(start=df.index[0], periods=35136, freq='15T'))
    df.set_index(newindex, inplace=True)

    # import weather data
    temp_url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/recent/10minutenwerte_TU_01270_akt.zip'
    r = requests.get(temp_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    temp = pd.read_csv(z.open(z.namelist()[0]), sep=';')
    temp['datetime']=pd.to_datetime(temp.MESS_DATUM, format='%Y%m%d%H%M')
    temp.set_index('datetime', inplace=True)
    temp.drop(['STATIONS_ID', 'MESS_DATUM', '  QN', 'PP_10', 'TM5_10', 'RF_10', 'TD_10', 'eor'], axis=1, inplace=True)
    temp = temp.resample("15T").mean()
    temp = temp.loc['2020',:]

    wind_url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/wind/recent/10minutenwerte_wind_01270_akt.zip'
    r = requests.get(wind_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    wind = pd.read_csv(z.open(z.namelist()[0]), sep=';')
    wind['datetime']=pd.to_datetime(wind.MESS_DATUM, format='%Y%m%d%H%M')
    wind.set_index('datetime', inplace=True)
    wind.drop(['STATIONS_ID', 'MESS_DATUM', '  QN', 'DD_10', 'eor'], axis=1, inplace=True)
    wind = wind.resample("15T").mean()
    wind = wind.loc['2020',:]

    sun_url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/recent/10minutenwerte_SOLAR_01270_akt.zip'
    r = requests.get(sun_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    sun = pd.read_csv(z.open(z.namelist()[0]), sep=';')
    sun['datetime']=pd.to_datetime(sun.MESS_DATUM, format='%Y%m%d%H%M')
    sun.set_index('datetime', inplace=True)
    sun.drop(['STATIONS_ID', 'MESS_DATUM', '  QN', 'DS_10', 'SD_10', 'LS_10', 'eor'], axis=1, inplace=True)
    sun = sun.resample("15T").mean()
    sun = sun.loc['2020',:]
    sun.loc[sun.GS_10==-999] = np.nan
    sun.iloc[0,0]=0
    sun = sun.fillna(method='ffill')

    # concatenate dfs:
    data = pd.concat([df, temp, sun, wind], axis=1)
    data.index.name = 'date_time'

    # normalize data
    scaler = StandardScaler()
    normed_data = pd.DataFrame(scaler.fit_transform(data.values), columns=data.columns, index=data.index)

    # set daily max of grid load to 1, everything else to 0 (label definitions)
    dailymax = data.groupby(pd.Grouper(freq='D'))['Wert'].idxmax()
    data['daily_max'] = 0
    data.loc[dailymax, 'daily_max'] = 1


    data.to_csv(os.path.join(FLAGS.experiment.file_path, 'data.csv'))

    normed_data['daily_max'] = data['daily_max']
    normed_data.to_csv(os.path.join(FLAGS.experiment.file_path, 'data_normed.csv'))

    print(normed_data.head())


if __name__ == '__main__':
    main()





