from copyreg import constructor
import os
import csv
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import twint
from os import path
from cgitb import reset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
from scipy.stats import pearsonr

	


#scrapes tweets, uncomment data line if you've never tried these dates before
def scrapeTweets(start_date, end_date, sentiment_correlation):
    delta = dt.timedelta(days=1)
    sentiment = []
    while start_date <= end_date:
        #string = 'twint --search "bitcoin, btc" --lang en --since "' + str((start_date - delta)) + ' 22:59:59" --until "' + str(start_date) + ' 22:59:59" -o outputs/' + str(start_date) + '.csv --csv'
        #os.system(string)
        sentimentAnalysis(start_date, start_date+delta, sentiment, sentiment_correlation)
        start_date += delta
    return sentiment

#actually performs sentiment analysis on tweets
def sentimentAnalysis(start_date, stop_date, sentiment, sentiment_correlation):
    #find and parse the csv file
    for file in os.listdir("./outputs"):
        if file.endswith(str(start_date)+".csv"):
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

def scrapeNews(start_date, end_date, sentiment_correlation):

    #set up the api 
    url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"

    querystring = {"q":"bitcoin","pageNumber":"1","pageSize":"50","autoCorrect":"true","fromPublishedDate":"null","toPublishedDate":"null"}

    headers = {
        'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
        'x-rapidapi-key': "d0eb600446mshf303aa8ca1b696dp10dd32jsnc6599e6c1cb5"
        }

    delta = dt.timedelta(days=1)
    pd.set_option('display.max_columns', None)
    while start_date <= end_date:
        
        start_date += delta
    
    exit()
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

    #initialize variables
    sentiment_correlation = []
    percentual_increase = []
    total = 0

    print("Would you like to use Twitter(T), news sources (N), quit (Q) ")
    choice = input()
    if(choice == 'T'):
        sentiment = scrapeTweets(start_date, end_date, sentiment_correlation)
    elif(choice == 'N'):
        sentiment = scrapeNews(start_date, end_date, sentiment_correlation)
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
		
    #plt.scatter(sentiment_correlation, percentual_increase)
    #plt.xlabel('sentiment')
    #plt.ylabel('percentage')
    #plt.grid(True)
    #plt.show()





