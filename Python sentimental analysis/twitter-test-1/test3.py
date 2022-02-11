import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import timezone
from datetime import datetime
import pandas as pd
import yfinance as yf


start = dt.datetime(2020,1,1)
end = dt.datetime.now()
print(start)
eth = yf.download('BTC-USD', start, end)
print(eth.Close)


# p_array = eth.open
# price = p_array[:,1]
# fig, ax = plt.subplots(1)
# fig.autofmt_xdate()
# plt.ylabel("Price") 
# plt.xlabel("Date and time") 
# plt.ylim(1000, 70000)
# plt.plot(time, price)
# plt.axvline(50)

# plt.locator_params(axis="x", nbins=5)


# plt.show()
