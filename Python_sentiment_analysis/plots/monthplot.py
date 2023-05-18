import plotly.express as px
import pandas as pd

df = pd.read_csv("monthPlot.csv")
fig = px.line(df, x = "months trained", y = "accuracy", color="model or dummy",color_discrete_map={"model":"red","dummy":"blue"}, title='Accuracy compared to months trained')
fig.write_image("sentBTC.png")
