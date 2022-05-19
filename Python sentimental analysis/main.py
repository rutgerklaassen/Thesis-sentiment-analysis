from __future__ import print_function

import os
import csv
from secrets import choice
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import twint
import time
import json
import numpy as np

from re import search
from urllib import response
from copyreg import constructor
from joblib import dump, load
from os import path
from cgitb import reset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier





#scrapes tweets, uncomment data line if you've never tried these dates before
def scrapeTweets(start_date, end_date, day_sentiment, choice1):
    delta = dt.timedelta(days=1)
    sentiment = []
    while start_date <= end_date:
        if(choice1 == 'Y'):
            string = 'twint --search "bitcoin, btc" --lang en --since "' + str((start_date - delta)) + ' 22:59:59" --until "' + str(start_date) + ' 22:59:59" -o twitteroutputs/' + str(start_date) + '.csv --csv'
            #os.system(string)
            sentimentAnalysis(start_date, start_date+delta, sentiment, day_sentiment, "twitter")
        else:
            readSentiment(start_date, day_sentiment, 'twitter')
        start_date += delta
    return sentiment


def sentimentAnalysis(start_date, stop_date, sentiment, day_sentiment, platform):
    #find and parse the csv file


    #put csv file data in an array
    if(platform == "twitter"):
        filepath = ""
        for file in os.listdir("./twitteroutputs"):
            if file.endswith(str(start_date)+".csv"):
                filepath = os.path.join("./twitteroutputs/", file)

        #reads csv
        data = pd.read_csv(filepath)
        posts = data['tweet'].values

    if(platform == "reddit"):
        filepath = ""
        for file in os.listdir("redditoutputs"):
            if file.endswith(str(start_date)+".csv"):
                filepath = os.path.join("./redditoutputs/", file)

        #reads csv
        data = pd.read_csv(filepath)
        posts = data['Text'].values
    totalscore = 0
    highestScore = 0
    lowestScore = 1
    numberOfPosts = 0
    #determine ploarity value by sentiment analysis
    for post in posts:   
        numberOfPosts += 1
        score = sid_obj.polarity_scores(post)['compound']
        if score > highestScore:
            highestScore = score
        if score < lowestScore:
            lowestScore = score
        totalscore += score

    
    if(len(posts)!=0):
        totalscore = totalscore / len(posts)
    day_sentiment.append(totalscore)
    highest_score.append(highestScore)
    lowest_score.append(lowestScore)
    number_of_posts.append(numberOfPosts)

    if(platform == 'reddit'):
        filename = 'redditSentimentScores/' + str(start_date)
        file = open(filename, "w")
        file.write(
            totalscore + "\n"
            + highestScore + "\n"
            + lowestScore + "\n"
            + numberOfPosts + "\n"
        )
        file.close()
    if(platform == 'twitter'):
        filename = 'twitterSentimentScores/' + str(start_date)
        file = open(filename, "w")
        file.write(
            str(totalscore) + "\n"
            + str(highestScore) + "\n"
            + str(lowestScore) + "\n"
            + str(numberOfPosts) + "\n"
        )
        file.close()   
    
    #add colours to array for graph
    if(totalscore > 0.1):
        sentiment.append('green')
    elif(totalscore < -0.1):
        sentiment.append('red')
    else:
        sentiment.append('gray')

def readSentiment(start_date, day_sentiment, platform):
    print('twitterSentimentScores/'+str(start_date))
    if(platform == 'twitter'):
        with open('twitterSentimentScores/'+str(start_date)) as file:
            lines = file.readlines()
            day_sentiment.append(lines[0])
            highest_score.append(lines[1])
            lowest_score.append(lines[2])
            number_of_posts.append(lines[3])
    if(platform == 'reddit'):
        with open('redditSentimentScores/'+str(start_date)) as file:
            lines = file.readlines()
            day_sentiment.append(lines[0])
            highest_score.append(lines[1])
            lowest_score.append(lines[2])
            number_of_posts.append(lines[3])

#calculate percentual increase per day and add to array
def calculateIncrease(percentual_increase, start, end):

    #get coin prices
    btc = yf.download('BTC-USD', start, end)
    btc = btc.reset_index()
    #enter coin data in global variables
    open = btc.Open
    close = btc.Close

    for idx, price in enumerate(open):
        difference = close[idx] - open[idx]
        percent = difference / open[idx] * 100
        percentual_increase.append(percent)
        if percent > 0:
            increase.append(1)
        else:
            increase.append(0)
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

def scrapeReddit(start_date, end_date, day_sentiment, choice1):
    #Subreddit to query
    subreddits =['CryptoCurrency', 'Crypto_Currency_News', 'CryptoMarkets', 'CryptoCurrencies']
    delta = dt.timedelta(days=1)
    sentiment = []
    while start_date <= end_date:
        if(choice1 == 'Y'):
            for sub in subreddits:
                before = int(time.mktime(dt.datetime.strptime(str(start_date+delta).replace('-','/'), "%Y/%m/%d").timetuple()))
                after =  int(time.mktime(dt.datetime.strptime(str(start_date).replace('-','/'), "%Y/%m/%d").timetuple()))
                query = "bitcoin"
                print(query, after, before, sub)
                data = getPushshiftData(query, after, before, sub)
                while len(data) > 0:
                    for submission in data:
                        collectSubData(submission)
                    # Calls getPushshiftData() with the created date of the last submission
                    after = data[-1]['created_utc']
                    data = getPushshiftData(query, after, before, sub)
            updateSubs_file(start_date)
            sentimentAnalysis(start_date, start_date+delta, sentiment, day_sentiment, "reddit")
        else:
            readSentiment(start_date, day_sentiment, 'twitter')
        start_date += delta
    return sentiment


def getPushshiftData(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url, headers = {"user-agent": "python:Sentiment_analysis_crypto:0.1 (by /u/edgarvandeplus)"})
    time.sleep(1)
    #print (r.text)
    if(r.text):
        try:
            data = json.loads(r.text)
        except:
            return None
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


def updateSubs_file(name):
    upload_count = 0
    filename = "redditoutputs/"+ str(name) + ".csv"
    file = filename
    with open(file, 'w', newline='', encoding='utf-8') as file: 
        a = csv.writer(file, delimiter=',')
        headers = ["Text"]
        a.writerow(headers)
        for sub in subStats:
            a.writerow(subStats[sub])
            upload_count+=1
            
        print(str(upload_count) + " submissions have been uploaded")

def train_support_vector(df_train, df_test, increase_train, increase_test, start_date, end_date, choice2):
    svc_model = SVC()
    scaler = StandardScaler().fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    #perform_prediction(svc_model.fit(df_train, increase_train)df_train, df_test, increase_train, increase_test, start_date, end_date, choice2)


def build_sets(day_sentiment, highest_score, lowest_score, number_of_posts, start_date, end_date, choice2):
    d = {'Sentiment': day_sentiment, 'Highest': highest_score, 'Lowest': lowest_score, 'Amount':number_of_posts}
    df = pd.DataFrame(d)
    modelName = "models/" + str(start_date) + "|" + str(end_date) + ".joblib"
    df_train, df_test, increase_train, increase_test = train_test_split(df,increase, train_size = 0.99, random_state=100)
    scaler = StandardScaler().fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    if(choice2 == 'S'):
        svc_model = ()
        model = svc_model.fit(df_train, increase_train)
        dump(model, modelName)
        increase_predict = model.predict(df)
        cm = np.array(confusion_matrix(increase, increase_predict, labels=[0,1]))
        confusion = pd.DataFrame(cm, index=['Increase', 'Decrease'], columns=['Predicted increase', 'Predicted Decrease'])
        print (confusion) 
        exit()
    else:
        print("What model would you like to load? startDate|EndDate")

        loadModel = input()
        model = load("models/" + loadModel + ".joblib")
        dummyModel = DummyClassifier()
        bl = dummyModel.fit(df_train, increase_train)
        baseline = bl.predict(df)
        dummyCm = np.array(confusion_matrix(increase, baseline, labels=[0,1]))
        dummyConfusion = pd.DataFrame(dummyCm, index=['Increase', 'Decrease'], columns=['Predicted increase', 'Predicted Decrease'])
        # print(dummyConfusion)
        # print(dummyCm)
        # print(increase)

        increase_predict = model.predict(df)
        
        cm = np.array(confusion_matrix(increase, increase_predict, labels=[0,1]))
        confusion = pd.DataFrame(cm, index=['Increase', 'Decrease'], columns=['Predicted increase', 'Predicted Decrease'])
        print (confusion) 
        exit()
# def perform_prediction(model, df_test, increase_test, start_date, end_date, choice2): 
#     if(choice2 == 'S'):      
#         #save model
#         dump(model, modelName)
#     else:
#         model = load(modelName)
#         increase_predict = model.predict(df_test)
#         cm = np.array(confusion_matrix(increase_test, increase_predict, labels=[0,1]))
#         confusion = pd.DataFrame(cm, index=['Increase', 'Decrease'], columns=['Predicted increase', 'Predicted Decrease'])
#         print (confusion)   

#     exit()

def getDates():
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
    return start, start_date, end, end_date

def menu(start_date, end_date, day_sentiment):
    print("Would you like to generate new sentiment values? (Y/N)")
    choice1 = input()
    print("Would you like to save a model or load it? (S/L)")
    choice2 = input()
    print("Would you like to use Twitter(T), Reddit (R), quit (Q) ")
    choice3 = input()
    if(choice3 == 'T'):
        sentiment = scrapeTweets(start_date, end_date, day_sentiment, choice1)
    elif(choice3 == 'R'):
        sentiment = scrapeReddit(start_date, end_date, day_sentiment, choice1)
    elif(choice3 == 'Q'):
        quit()
    return choice2

def main():
    start, start_date, end, end_date = getDates()

    choice2 = menu(start_date, end_date, day_sentiment)

    calculateIncrease(percentual_increase, start, end)

    build_sets(day_sentiment, highest_score, lowest_score, number_of_posts, start_date, end_date, choice2)

    # print("setnimetnt ::: ")
    # print(day_sentiment)
    # total = sentimentAccuracy(sentiment, open, close, total)
		
    # print(total)
    # print(len(sentiment))
    # print(total/len(sentiment), "%")

    # corr, _ = pearsonr(day_sentiment, percentual_increase)
    # print('Spearmans correlation: %.3f' % corr)
    # plot the price graph with selling points
    # plt.plot(dates, open)
    # plt.xlabel('datum')
    # plt.ylabel('prijs')
    # plt.scatter(dates, open, s = 50, c = sentiment)
    # plt.grid (True)
    # plt.show()
    # plt.close()
    
    #plot the scatter plot for correlation between sentiment and price increase
		
    # plt.scatter(day_sentiment, percentual_increase)
    # plt.xlabel('sentiment')
    # plt.ylabel('percentage')
    # plt.grid(True)
    # plt.show()
if __name__ == "__main__":	
    #initialize sentiment object
    sid_obj = SentimentIntensityAnalyzer()

    #initialize global variables
    day_sentiment = []
    highest_score = []
    lowest_score = []
    number_of_posts = []

    percentual_increase = []
    increase = []
    
    total = 0
    subStats = {}


    main()




