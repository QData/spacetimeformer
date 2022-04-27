
#!/bin/sh
echo "acquiring up-to-date data from Binance"
rm Binance_ETHUSDT_minute.csv
rm Binance_BTCUSDT_minute.csv
wget https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_minute.csv https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_minute.csv --no-check-certificate
sed -i '1d' Binance_ETHUSDT_minute.csv
sed -i '1d' Binance_BTCUSDT_minute.csv
python ConvertData.py