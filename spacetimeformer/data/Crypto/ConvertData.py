import pandas as pd
import requests
from datetime import datetime

BTCUSDT = pd.read_csv('BTCUSDT_Binance_futures_data_hour.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
ETHUSDT = pd.read_csv('ETHUSDT_Binance_futures_data_hour.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
LINKUSDT = pd.read_csv('LINKUSDT_Binance_futures_data_hour.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
EOSUSDT = pd.read_csv('EOSUSDT_Binance_futures_data_hour.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
XMRUSDT = pd.read_csv('XMRUSDT_Binance_futures_data_hour.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
NEOUSDT = pd.read_csv('NEOUSDT_Binance_futures_data_hour.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
LTCUSDT = pd.read_csv('LTCUSDT_Binance_futures_data_hour.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)



ETHUSDT.rename(columns = {'open': 'ETH_open', 'high': 'ETH_high', 'low': 'ETHT_low', 'close': 'ETH_close', 'volume': 'ETH_volume', 'tradecount': 'ETH_tradecount', 'date' :'Datetime'}, inplace = True)
BTCUSDT.rename(columns = {'open': 'BTC_open', 'high': 'BTC_high', 'low': 'BTC_low', 'close': 'BTC_close', 'volume': 'BTC_volume', 'tradecount': 'BTC_tradecount','date' :'Datetime'}, inplace = True)
LINKUSDT.rename(columns = {'open': 'LINK_open', 'high': 'LINK_high', 'low': 'LINK_low', 'close': 'LINK_close', 'volume': 'LINK_volume', 'tradecount': 'LINK_tradecount','date' :'Datetime'}, inplace = True)
EOSUSDT.rename(columns = {'open': 'EOS_open', 'high': 'EOS_high', 'low': 'EOS_low', 'close': 'EOS_close', 'volume': 'EOS_volume', 'tradecount': 'EOS_tradecount','date' :'Datetime'}, inplace = True)
XMRUSDT.rename(columns = {'open': 'XMR_open', 'high': 'XMR_high', 'low': 'XMR_low', 'close': 'XMR_close', 'volume': 'XMR_volume', 'tradecount': 'XMR_tradecount','date' :'Datetime'}, inplace = True)
NEOUSDT.rename(columns = {'open': 'NEO_open', 'high': 'NEO_high', 'low': 'NEO_low', 'close': 'NEO_close', 'volume': 'NEO_volume', 'tradecount': 'NEO_tradecount','date' :'Datetime'}, inplace = True)
LTCUSDT.rename(columns = {'open': 'LTC_open', 'high': 'LTC_high', 'low': 'LTC_low', 'close': 'LTC_close', 'volume': 'LTC_volume', 'tradecount': 'LTC_tradecount','date' :'Datetime'}, inplace = True)

# LTCUSDT.rename(columns = {'open': 'LTC_open', 'high': 'LTC_high', 'low': 'LTC_low', 'close': 'LTC_close', 'volume': 'LTC_volume', 'tradecount': 'LTC_tradecount', 'date' :'Datetime'}, inplace = True)
df = pd.concat([ETHUSDT, BTCUSDT,LINKUSDT, EOSUSDT, XMRUSDT, NEOUSDT, LTCUSDT], axis = 1)
#Remove all duplicate units and date column_set
df = df.loc[:,~df.columns.duplicated()]
# remove symbol column
df = df.drop(columns = ['symbol',"unix"])

api_url = "https://api.alternative.me/fng/?limit=0&date_format=us"
raw_df = requests.get(api_url).json()


timestamp = []
value = []
value_classification = []
for i in raw_df["data"]:
  timestamp.append( i["timestamp"])
  value.append(int(i["value"]))
  value_classification.append(i["value_classification"])
sm = pd.DataFrame([timestamp,value]).T
sm.columns = ["Datetime","sentiment",]


sm["Datetime"] = sm["Datetime"].apply(lambda d: datetime.strptime(d, "%m-%d-%Y"))
sm.index = sm.Datetime

sntiments= []
for index, row in df.iterrows():
  date = row["Datetime"].strftime('%Y-%m-%d')
  s = sm[sm['Datetime'].dt.strftime('%Y-%m-%d') == date]
  sntiments.append(s["sentiment"].values[0])
df["sntiments"] = sntiments
count = len(df)
df.dropna()

print("cleaning data said total rows left", count, "total rows left", len(df))
print("saving data to csv")
df.to_csv('../crypto_dset.csv', index = False)
print("done")
print(df.columns)
print(len(df.columns))