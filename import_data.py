import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()
db_password = os.getenv("DB_PASSWORD")

# SQLAlchemy connection string
engine = create_engine(f'mysql+pymysql://root:{db_password}@localhost/flood_disaster_db')

def load_text_file(filename, value_name):
    """Load raw text data and parse datetime/numeric values"""
    df = pd.read_csv(filename, skiprows=9, header=None, names=['date', 'time', value_name, 'flag'])
    
    # Handling '24:00' format and converting to datetime
    df['observation_datetime'] = df.apply(
        lambda x: pd.to_datetime(x['date'] + ' 00:00') + pd.Timedelta(days=1) 
        if x['time'] == '24:00' else pd.to_datetime(x['date'] + ' ' + x['time']), axis=1
    )
    
    df[value_name] = pd.to_numeric(df[value_name], errors='coerce')
    return df[['observation_datetime', value_name]]

# Data Integration Process
df_level = load_text_file('kinugawamitsukaido_waterlevel.txt', 'water_level_m')
df_rain_l = load_text_file('mitsukaido_rain.txt', 'rain_local_mm')
df_rain_u = load_text_file('kawaji_rain.txt', 'rain_upstream_mm')

df_merged = pd.merge(df_level, df_rain_l, on='observation_datetime', how='inner')
df_merged = pd.merge(df_merged, df_rain_u, on='observation_datetime', how='inner')

# Upsert into MySQL
df_merged.to_sql('kinugawa_hydromet', con=engine, if_exists='append', index=False)
