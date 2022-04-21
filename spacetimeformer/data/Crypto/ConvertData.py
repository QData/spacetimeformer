import pandas as pd
ETHUSDT = pd.read_csv('spacetimeformer/data/Crypto/Binance_BTCUSDT_minute.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
BTCUSDT = pd.read_csv('spacetimeformer/data/Crypto/Binance_BTCUSDT_minute.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)
LTCUSDT = pd.read_csv('spacetimeformer/data/Crypto/Binance_LTCUSDT_minute.csv', parse_dates = ['date']).sort_values(by = 'date').reset_index(drop = True)

# rename each column and tag it with its symbols
ETHUSDT.rename(columns = {'open': 'ETH_open', 'high': 'ETH_high', 'low': 'ETHT_low', 'close': 'ETH_close', 'volume': 'ETH_volume', 'tradecount': 'ETH_tradecount'}, inplace = True)
# now repeat this for all datasets
BTCUSDT.rename(columns = {'open': 'BTC_open', 'high': 'BTC_high', 'low': 'BTC_low', 'close': 'BTC_close', 'volume': 'BTC_volume', 'tradecount': 'BTC_tradecount'}, inplace = True)
LTCUSDT.rename(columns = {'open': 'LTC_open', 'high': 'LTC_high', 'low': 'LTC_low', 'close': 'LTC_close', 'volume': 'LTC_volume', 'tradecount': 'LTC_tradecount'}, inplace = True)


df = pd.concat([ETHUSDT, BTCUSDT, LTCUSDT], axis = 1)

#Remove all duplicate units and date column_set
df = df.loc[:,~df.columns.duplicated()]
# remove symbol column
df = df.drop(columns = ['symbol'])

df.to_csv('spacetimeformer/data/Crypto/crypto_converted.csv', index = False)
