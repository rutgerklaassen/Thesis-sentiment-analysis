from __future__ import print_function

import os
import csv
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import requests
import twint
import time
import json
import sys
import numpy as np
from re import M, search
from urllib import response
from copyreg import constructor

from os import path
from cgitb import reset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
from scipy.stats import pearsonr, skew, kurtosis
from googleapiclient.discovery import build
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot
from joblib import dump, load

if __name__ == "__main__":


    start_date = sys.argv[1]
    print("start date is : ", start_date)
    start_date = start_date.split("-")
    start = dt.datetime(int(start_date[0]),int(start_date[1]),int(start_date[2]))
    start_date = dt.date(int(start_date[0]),int(start_date[1]),int(start_date[2]))

    end_date = sys.argv[2]
    print("end date is : ", end_date)
    end_date = end_date.split("-")
    end = dt.datetime(int(end_date[0]),int(end_date[1]),int(end_date[2]))
    end_date = dt.date(int(end_date[0]),int(end_date[1]),int(end_date[2]))


    delta = dt.timedelta(hours=1)
    timezoneCorrector = dt.timedelta(hours=6)
    dateTime = start
    while dateTime < end:
        removespace = str(dateTime - timezoneCorrector).replace(" ","H").replace(":00:00","")
        string = 'twint --search "dogecoin, doge" --lang en --since "' + str((dateTime)) + '" --until "' + str(dateTime + delta) + '" -o outputs/' + removespace + '.csv --csv'
        print(string)
        os.system(string)
        dateTime += delta
