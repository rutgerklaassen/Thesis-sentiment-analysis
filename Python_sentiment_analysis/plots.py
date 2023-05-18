import os
import csv
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import plotly.express as px

if __name__ == "__main__":

    sentiment = []
    increase = []
    directory = 'sentimentOutputs'
    for file in os.listdir(directory):
        filepath = os.path.join("./sentimentOutputs/", file)
        if file.endswith(".csv"):
            print(filepath)   
            df = pd.read_csv(filepath)
            df.columns = [c.replace(' ', '_') for c in df.columns]
            sentiment.append(df.day_sentiment[0])
            increase.append(df.percentual_increase[0])
    df = pd.DataFrame({'sentiment':sentiment, 'price change (%)':increase})
    print(increase)
    print(sentiment)
    fig = px.scatter(x = sentiment, y = increase)
    fig.update_layout(
        title="Sentiment score compared to percentual price change",
        xaxis_title="sentiment",
        yaxis_title="price change (%)",
    )

fig.write_image("scatterXRP.png")


