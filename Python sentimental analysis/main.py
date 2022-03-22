from __future__ import print_function

import os
import csv
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import twint
import time
import json

from re import search
from urllib import response
from copyreg import constructor

from os import path
from cgitb import reset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
from scipy.stats import pearsonr
from googleapiclient.discovery import build



#scrapes tweets, uncomment data line if you've never tried these dates before
def scrapeTweets(start_date, end_date, sentiment_correlation):
    delta = dt.timedelta(days=1)
    sentiment = []
    while start_date <= end_date:
        #string = 'twint --search "bitcoin, btc" --lang en --since "' + str((start_date - delta)) + ' 22:59:59" --until "' + str(start_date) + ' 22:59:59" -o outputs/' + str(start_date) + '.csv --csv'
        #os.system(string)
        sentimentAnalysis(start_date, start_date+delta, sentiment, sentiment_correlation, "twitter")
        start_date += delta
    return sentiment

#actually performs sentiment analysis on tweets
def sentimentAnalysisTwitter(start_date, stop_date, sentiment, sentiment_correlation):
    #find and parse the csv file
    filepath = ""
    for file in os.listdir("./outputs"):
        #if file.endswith(str(start_date)+".csv"):
        if file.endswith("test3"+".csv"):
            filepath = os.path.join("./outputs/", file)

    #reads csv
    data = pd.read_csv(filepath)

    #put csv file data in an array
    tweets = []
    tweets = data['tweet'].values
    totalscore = 0

    #determine ploarity value by sentiment analysis
    for tweet in tweets:   
        totalscore += sid_obj.polarity_scores(tweet)['compound']

    
    if(len(tweets)!=0):
        totalscore = totalscore / len(tweets)
    sentiment_correlation.append(totalscore)

    #add colours to array for graph
    if(totalscore > 0.1):
        sentiment.append('green')
    elif(totalscore < -0.1):
        sentiment.append('red')
    else:
        sentiment.append('gray')

def sentimentAnalysis(start_date, stop_date, sentiment, sentiment_correlation, platform):
    #find and parse the csv file
    for file in os.listdir("./outputs"):
        if file.endswith(str(start_date)+".csv"):
        #if file.endswith("test3"+".csv"):
            filepath = os.path.join("./outputs/", file)

    #reads csv
    data = pd.read_csv(filepath)

    #put csv file data in an array
    if(platform == "twitter"):
        posts = data['tweet'].values
        totalscore = 0

    if(platform == "reddit"):
        posts = data['Text'].values
        totalscore = 0
        
    #determine ploarity value by sentiment analysis
    for post in posts:   
        totalscore += sid_obj.polarity_scores(post)['compound']

    
    if(len(posts)!=0):
        totalscore = totalscore / len(posts)
    sentiment_correlation.append(totalscore)

    #add colours to array for graph
    if(totalscore > 0.1):
        sentiment.append('green')
    elif(totalscore < -0.1):
        sentiment.append('red')
    else:
        sentiment.append('gray')
#calculate percentual increase per day and add to array
def calculateIncrease(percentual_increase):
    for idx, price in enumerate(open):
        difference = close[idx] - open[idx]
        increase = difference / open[idx] * 100
        percentual_increase.append(increase)
    return percentual_increase

#check if the sentiment is accurate for predicting the price
def sentimentAccuracy(sentiment, open, close, total):
    for idx, colour in enumerate(sentiment):
        print(idx, colour, close[idx], open[idx])
        if(colour == "green"):
            if(close[idx]>open[idx] * 1.01):
                total = total + 100
                print("juist")
        elif(colour == "red"):
            if(close[idx]<open[idx] * 0.99):
                total = total + 100
                print("juist")
        else:
            if(close[idx]<(open[idx] * 1.01) and 
            close[idx]>(open[idx] * 0.99)):
                print("juist")
                total = total + 100
    return total

# def scrapeYoutube(start_date, end_date, sentiment_correlation):

#     #set up the api 
#     api_key = 'AIzaSyD39VCOVHS_ljckZFpOdN2ZfIsXS9EfhLg'
#     youtube = build('youtube', 'v3', developerKey=api_key)

    
#     locations = ['US','GB','CA']
#     delta = dt.timedelta(days=1)
#     pd.set_option('display.max_columns', None)
#     titles = list(range(0))
#     while start_date <= end_date:
#         for countryCode in locations:
#             search_response = youtube.search().list(
#                 q='"bitcoin" | "BTC"',
#                 part = 'snippet',
#                 type = 'video',
#                 maxResults = 50,
#                 publishedAfter = str(start_date) + "T00:00:00Z",
#                 publishedBefore = str(start_date + delta) + "T00:00:00Z",
#                 regionCode = countryCode
#             )
#             response = search_response.execute()
#             videos = response['items']
#             for video in videos:
#                 titles.append(video['snippet']['title'].replace("&","").replace("#","").replace(";","").replace("amp",""))
#         toCsv(titles)    
#         start_date += delta
    
# def toCsv(titles):
#     zip(titles)
#     header = ["title"]
#     with open('countries.csv', 'w', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         for val in titles:
#             writer.writerow([val])

def scrapeReddit(start_date, end_date, sentiment_correlation):
    subreddits =['CryptoCurrency', 'Crypto_Currency_News', 'CryptoMarkets', 'CryptoCurrencies']
    subStats = {}
    delta = dt.timedelta(days=1)
    sentiment = []
    while start_date <= end_date:
        for sub in subreddits:
            before = "1609628400" 
            after = "1609542000"  
            query = "bitcoin"
            print(query, after, before, sub)
            data = getPushshiftData(query, after, before, sub)
            while len(data) > 0:
                for submission in data:
                    collectSubData(submission)
                # Calls getPushshiftData() with the created date of the last submission
                after = data[-1]['created_utc']
                data = getPushshiftData(query, after, before, sub)

    return sentiment

def getPushshiftData(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    #print(data['data'])
    return data['data']

def collectSubData(subm):
    subData = list() #list to store data points
    if 'selftext' in subm.keys():
        title = subm['title']  
        text = subm['selftext']
        sub_id = subm['id']
        if(text != '[removed]' and text != '[deleted]' and text!=''):
            subData.append((text))
            subStats[sub_id] = subData
        elif(title != '[removed]' and title != '[deleted]' and title != ''):
            subData.append((title))
            subStats[sub_id] = subData
        return

def updateSubs_file():
    print("binnen")
    upload_count = 0
    filename = input()
    file = filename
    with open(file, 'w', newline='', encoding='utf-8') as file: 
        a = csv.writer(file, delimiter=',')
        headers = ["Text"]
        a.writerow(headers)
        for sub in subStats:
            a.writerow(subStats[sub])
            upload_count+=1
            
        print(str(upload_count) + " submissions have been uploaded")


if __name__ == "__main__":	

    #initialize sentiment object
    sid_obj = SentimentIntensityAnalyzer()

    #enter correct dates
    print("enter start date (yyyy-mm-dd) : ")
    start_date = input()
    start_date = start_date.split("-")
    start = dt.datetime(int(start_date[0]),int(start_date[1]),int(start_date[2]))
    start_date = dt.date(int(start_date[0]),int(start_date[1]),int(start_date[2]))
    print("enter end date (yyyy-mm-dd) : ")
    end_date = input()
    end_date = end_date.split("-")
    end = dt.date(int(end_date[0]),int(end_date[1]),int(end_date[2]))
    end_date = dt.date(int(end_date[0]),int(end_date[1]),int(end_date[2]))

    #initialize global variables
    sentiment_correlation = []
    percentual_increase = []
    total = 0

    print("Would you like to use Twitter(T), Youtube (Y), quit (Q) ")
    choice = input()
    if(choice == 'T'):
        sentiment = scrapeTweets(start_date, end_date, sentiment_correlation)
    #elif(choice == 'Y'):
        #sentiment = scrapeYoutube(start_date, end_date, sentiment_correlation)
    elif(choice == 'R'):
        sentiment = scrapeReddit(start_date, end_date, sentiment_correlation)
    elif(choice == 'Q'):
        quit()
    print(sentiment)

    #get coin prices
    btc = yf.download('BTC-USD', start, end)
    btc = btc.reset_index()
    #enter coin data in variables
    open = btc.Open
    dates = btc.Date
    close = btc.Close
    

    calculateIncrease(percentual_increase)
    print("setnimetnt ::: ")
    print(sentiment_correlation)
    total = sentimentAccuracy(sentiment, open, close, total)
		
    print(total)
    print(len(sentiment))
    print(total/len(sentiment), "%")

    corr, _ = pearsonr(sentiment_correlation, percentual_increase)
    print('Spearmans correlation: %.3f' % corr)
    #plot the price graph with selling points
    # plt.plot(dates, open)
    # plt.xlabel('datum')
    # plt.ylabel('prijs')
    # plt.scatter(dates, open, s = 50, c = sentiment)
    # plt.grid (True)
    # plt.show()
    # plt.close()
    
    #plot the scatter plot for correlation between sentiment and price increase
		
    plt.scatter(sentiment_correlation, percentual_increase)
    plt.xlabel('sentiment')
    plt.ylabel('percentage')
    plt.grid(True)
    plt.show()





