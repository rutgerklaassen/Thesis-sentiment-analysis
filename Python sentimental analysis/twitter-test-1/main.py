import os
import csv
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Scweet.scweet import scrape
from Scweet.user import get_user_information, get_users_following, get_users_followers
import yfinance as yf
	



def scrapeTweets(start_date, end_date):
    delta = dt.timedelta(days=1)
    sentiment = []
    while start_date <= end_date:
        #data = scrape(words=['bitcoin','ethereum'], since=str(start_date), until=str(start_date+delta), from_account = None, interval=1, headless=False, display_type="Top", save_images=False, lang="en",resume=False, filter_replies=False, proximity=False)
        sentimentAnalysis(start_date, start_date+delta, sentiment, sentiment correlation)
        start_date += delta
    return sentiment

def sentimentAnalysis(start_date, stop_date, sentiment):
    #find and parse the csv file
    print(sentiment)
    for file in os.listdir("./outputs"):
        if file.endswith(str(stop_date)+".csv"):
            filepath = os.path.join("./outputs/", file)
    #file = open(filepath)
    data = pd.read_csv(filepath)
    #put csv file data in an array
    tweets = []
    tweets = data['Embedded_text'].values
    totalscore = 0
    #print(start_date)
    for tweet in tweets:
        totalscore += sid_obj.polarity_scores(tweet)['compound']
    if(len(tweets)!=0):
    	totalscore = totalscore / len(tweets)
    print (totalscore)
    if(totalscore > 0.1):
        sentiment.append('green')
    elif(totalscore < -0.1):
        sentiment.append('red')
    else:
        sentiment.append('gray')
    
    



if __name__ == "__main__":	
    sid_obj = SentimentIntensityAnalyzer()
    #enter correct dates
    print("start date (yyyy-mm-dd) : ")
    start_date = input()
    start_date = start_date.split("-")
    start = dt.datetime(int(start_date[0]),int(start_date[1]),int(start_date[2]))
    start_date = dt.date(int(start_date[0]),int(start_date[1]),int(start_date[2]))
    print("until (yyyy-mm-dd) : ")
    end_date = input()
    end_date = end_date.split("-")
    end = dt.date(int(end_date[0]),int(end_date[1]),int(end_date[2]))
    end_date = dt.date(int(end_date[0]),int(end_date[1]),int(end_date[2]))

    
    sentiment = scrapeTweets(start_date, end_date)
    print(sentiment)
	#start scraping tweets
    print(start)
    btc = yf.download('BTC-USD', start, end)
    btc = btc.reset_index()
    print(btc)  
    print(btc.Date)
    open = []
    open = btc.Open
    dates = btc.Date
    close = btc.Close
    total = 0
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

    print(total)
    print(len(sentiment))
    print(total/len(sentiment), "%")

    
    plt.plot(dates, open)
    plt.xlabel('datum')
    plt.ylabel('prijs')
    plt.scatter(dates, open, s = 50, c = sentiment)
    plt.grid (True)
    plt.show()






    # test =+ sid_obj.polarity_scores("Idiots are people who are less intelligent")['compound']
    # letters = ["fuck","jesus christ","fucking hell","love"]
    # totalscore = 0
    # for tweet in letters:
    #     print(tweet)
    #     print(sid_obj.polarity_scores(tweet)['compound'])
    #     totalscore += sid_obj.polarity_scores(tweet)['compound']
    # print(totalscore)
    # print(totalscore / len(letters))
    #print(test)
    #print(sentiment_dict)
