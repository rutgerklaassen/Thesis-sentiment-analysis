import os
import csv
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import re

def replaceNonAlpha(element): #replaces all non alphabetic char except spaces
    element = re.sub(r'[^A-Za-z ]+', '', element)
    return element

if __name__ == "__main__":


    directory = 'outputs'
    for file in os.listdir(directory):
        filepath = os.path.join("./outputs/", file)
        if file.endswith(".csv"):
            print(filepath)   
            df = pd.read_csv(filepath, sep="\t")
            print(df)
            try:
                df.loc[:,'tweet'] = [replaceNonAlpha(x) for x in df.tweet]
                df = df.drop_duplicates(subset='tweet', keep="first")
                df.to_csv(filepath)
            except:
                print("oh nee")




