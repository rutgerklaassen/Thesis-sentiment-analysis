from pycoingecko import CoinGeckoAPI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import timezone
from datetime import datetime
import numpy as np
import altair as alt
import pandas as pd
import altair_viewer

def datetime_to_unix(year, month, day):
    '''datetime_to_unix(2021, 6, 1) => 1622505600.0'''
    dt = datetime(year, month, day)
    timestamp = (dt - datetime(1970, 1, 1)).total_seconds()
    return timestamp
def unix_to_datetime(unix_time):
    '''unix_to_datetime(1622505700)=> ''2021-06-01 12:01am'''
    ts = int(unix_time/1000 if len(str(unix_time)) > 10 else unix_time) # /1000 handles milliseconds
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M').lower()

# Initialize the client
cg = CoinGeckoAPI()
# Retrieve Bitcoin data in USD
result = cg.get_coin_market_chart_range_by_id(
    id='bitcoin',
    vs_currency='usd',
    from_timestamp=datetime_to_unix(2021, 6, 1),
    to_timestamp=datetime_to_unix(2021, 6, 8)
)


time = [ unix_to_datetime(i[0]) for i in result['prices'] ]

p_array = np.array(result['prices'])
price = p_array[:,1]
fig, ax = plt.subplots(1)
fig.autofmt_xdate()
plt.ylabel("Price") 
plt.xlabel("Date and time") 
plt.ylim(1000, 70000)
plt.plot(time, price)
plt.axvline(50)

plt.locator_params(axis="x", nbins=5)


plt.show()
