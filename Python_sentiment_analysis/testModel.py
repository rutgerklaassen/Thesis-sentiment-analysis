from __future__ import print_function

import os
import datetime as dt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
import sys
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot
from joblib import dump, load


def sentimentAnalysis(date, sentiment, day_sentiment, hour):
    #find and parse the csv file


    #put csv file data in an array
    filepath = "" 
    dateString = str(date.year)

    #if statements to fix naming mistake with data filenames where the 0 isn't attached
    if(date.month<10):
        dateString+= "-0"+str(date.month)
    else:
        dateString+= "-"+str(date.month)
    if(date.day<10):
        dateString+= "-0"+str(date.day)
    else:
        dateString+= "-"+str(date.day)
    if(hour < 10):
        hour = "0" + str(hour)
    else:
        hour = str(hour)
    
    print("searching for :", dateString + "H" + hour + ".csv")
    for file in os.listdir("./outputs"):
        if file.endswith(dateString + "H" + hour + ".csv"):
            filepath = os.path.join("./outputs/", file)
        
    #reads csv
    try :
        posts = pd.read_csv(filepath, usecols=["tweet"], sep =',')
        #removes index and puts tweets in array
        posts = posts.tweet
    except :
        print("gaat iets fout bij deze filepath :", filepath)
        return

    totalscore = 0
    highestScore = -1
    lowestScore = 1
    numberOfPosts = 0
    price = -1
    nextPrice = -1
    nextDate = date + dt.timedelta(hours=1)
    #dateTime = dt.datetime.combine(date, dt.time(hour, 0, 0))

    #bug dissapeared but keep here just to be sure
    # if(hour < 22):
    #     nextDateTime = dt.datetime.combine(date, dt.time(hour + 1, 0, 0))
    # else:    #bug with yfinance where it doesn't work for 23
    #     print("1")
    #     return
    # if(coin.Close[dateTime] == -1 or coin.Close[nextDateTime] == -1):
    #     print("2")
    #     return
            

    #determine ploarity value by sentiment analysis
    scores = []
    for post in posts:   
        numberOfPosts += 1
        score = sid_obj.polarity_scores(post)['compound']
        scores.append(score)
        if score > highestScore:
            highestScore = score
        if score < lowestScore:
            lowestScore = score
        totalscore += score

    scores = np.array(scores)
    
    if(len(posts)!=0):
        totalscore = totalscore / len(posts)
    else:
        return
    
    day_sentiment.append(totalscore)
    highest_score.append(highestScore)
    lowest_score.append(lowestScore)
    number_of_posts.append(numberOfPosts)
    standard_deviations.append(scores.std())
    skewness.append(skew(scores))
    kurtosisList.append(kurtosis(scores))

    diff = coin.Close[nextDate] - coin.Close[dateTime]
    if(diff != 0):
        percent = diff / coin.Close[dateTime] * 100
        percentual_increase.append(percent)
    if(coin.Close[nextDate] > coin.Close[dateTime]):
        increase.append(1)

    else:
        increase.append(0)


    #add colours to array for graph
    if(totalscore > 0.1):
        sentiment.append('green')
    elif(totalscore < -0.1):
        sentiment.append('red')
    else:
        sentiment.append('gray')
#calculate percentual increase per day and add to array

def train_support_vector(df_train, df_test, increase_train, increase_test):
    svc_model = RandomForestClassifier(n_estimators=10000)
    scaler = StandardScaler().fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)

    #create dummy model 
    dm = DummyClassifier()
    bl = dm.fit(df_train, increase_train)
    perform_prediction(df_test, increase_test, bl)


def build_sets(day_sentiment, highest_score, lowest_score, number_of_posts, standard_deviations, skewness, kurtosisList):
    print(len(day_sentiment), len(percentual_increase))
    pearson = corr,_= pearsonr(day_sentiment, percentual_increase)
    # scatterplot()
    print('Pearsons correlation for sentiment : %.3f' % corr)
    #pyplot.scatter(day_sentiment, percentual_increase)
    d = {'Sentiment': day_sentiment, 'Highest': highest_score, 'Lowest': lowest_score, 'Amount':number_of_posts}
    df = pd.DataFrame(d)
    df_train, df_test, increase_train, increase_test = train_test_split(df,increase, test_size = 0.99, random_state=53)
    train_support_vector(df_train, df_test, increase_train, increase_test)

def perform_prediction(df_test, increase_test, bl):
    
    model = load(modelName)
    increase_predict = model.predict(df_test)
    #calculate baseline %
    baseline = bl.predict(df_test)
    dummyCm = np.array(confusion_matrix(increase_test, baseline, labels=[0,1]))
    total = dummyCm[0][0] + dummyCm[0][1] + dummyCm[1][0] + dummyCm[1][1]
    dummyPercentage = ((dummyCm[0][0] + dummyCm[1][1])/ total) * 100
    print("dummy : ", dummyPercentage)
    
    cm = np.array(confusion_matrix(increase_test, increase_predict, labels=[0,1]))
    confusion = pd.DataFrame(cm, index=['Increase', 'Decrease'], columns=['Predicted increase', 'Predicted Decrease'])
    print (confusion)
    total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
    percentage = ((cm[0][0] + cm[1][1])/ total) * 100
    print("real percentage : ", percentage)

    with open(str(start_date)+'||'+str(end_date)+"||"+modelName,'w') as file :
        file.write('starting at date : ' + str(start_date))
        file.write('\nending at date : ' + str(end_date))
        file.write('\nUsing model : ' + modelName)
        file.write('\nIncrease correct : ')
        file.write(str(cm[0][0]))
        file.write('\nIncrease incorrect : ')
        file.write(str(cm[0][1]))
        file.write('\nDecrease correct : ')
        file.write(str(cm[1][0]))
        file.write('\nDecrease incorrect : ')
        file.write(str(cm[1][1]))
        file.write('\nPearson\'s correlation : ' + str(float(pearson)))
        file.write('\ndummy percentage : ' + str(dummyPercentage) + '%')
        file.write('\nModel percentage : '+ str(percentage) + '%')


    exit()

if __name__ == "__main__":

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    #initialize sentiment object
    sid_obj = SentimentIntensityAnalyzer()

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

    modelName = sys.argv[3]
    
    #initialize global variables
    day_sentiment = []
    highest_score = []
    lowest_score = []
    number_of_posts = []
    standard_deviations = []
    skewness = []
    kurtosisList = []
    sentiment = []
    percentual_increase = []
    increase = []
    pearson = 0
    
    total = 0
    subStats = {}

    #the yfinance code is messy and you have to correct for which timezone you're in, this one is for america timezone
    timezoneCorrector = dt.timedelta(hours=6) 
    coin = yf.download('BTC-USD', (start-timezoneCorrector), end + dt.timedelta(hours=-4), interval="1h" ).reset_index()
    coin.rename(columns={coin.columns[0]: "dates"}, inplace = True)
    coin.set_index('dates', inplace=True)
    coin = coin.tz_localize(None)

    delta = dt.timedelta(hours=1)
    dateTime = start
    while dateTime < end:
        removespace = str(dateTime - timezoneCorrector).replace(" ","H").replace(":00:00","")
        sentimentAnalysis(dateTime, sentiment, day_sentiment, dateTime.hour)
        dateTime += delta

    build_sets(day_sentiment, highest_score, lowest_score, number_of_posts, standard_deviations, skewness, kurtosisList)
