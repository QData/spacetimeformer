
#!/bin/sh
echo "acquiring up-to-date data from Binance"
!rm BTCUSDT_Binance_futures_data_hour.csv
!rm ETHUSDT_Binance_futures_data_hour.csv
!rm LINKUSDT_Binance_futures_data_hour.csv
!rm EOSUSDT_Binance_futures_data_hour.csv
!rm XMRUSDT_Binance_futures_data_hour.csv
!wget https://www.cryptodatadownload.com/cdd/BTCUSDT_Binance_futures_data_hour.csv \
https://www.cryptodatadownload.com/cdd/ETHUSDT_Binance_futures_data_hour.csv \
https://www.cryptodatadownload.com/cdd/LINKUSDT_Binance_futures_data_hour.csv \
https://www.cryptodatadownload.com/cdd/EOSUSDT_Binance_futures_data_hour.csv \
https://www.cryptodatadownload.com/cdd/XMRUSDT_Binance_futures_data_hour.csv \
--no-check-certificate
!sed -i '1d' BTCUSDT_Binance_futures_data_hour.csv
!sed -i '1d' ETHUSDT_Binance_futures_data_hour.csv
!sed -i '1d' LINKUSDT_Binance_futures_data_hour.csv
!sed -i '1d' EOSUSDT_Binance_futures_data_hour.csv
!sed -i '1d' XMRUSDT_Binance_futures_data_hour.csv
python ConvertData.py