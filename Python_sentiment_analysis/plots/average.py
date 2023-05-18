import plotly.express as px
import pandas as pd

df = pd.read_csv("monthPlot.csv")
modelTotal = 0
modelAmount = 0
dummyTotal = 0
dummyAmount = 0
for index, row in df.iterrows():
    if(row['model or dummy']=='model'):
        modelTotal += row['accuracy']
        modelAmount += 1
    if(row['model or dummy']=='dummy'):
        dummyTotal += row['accuracy']
        dummyAmount += 1
print("average model = ", modelTotal/modelAmount)
print("average dummy = ", dummyTotal/dummyAmount)