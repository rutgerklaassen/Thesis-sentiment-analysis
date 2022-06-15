import sys
import datetime as dt
from tracemalloc import start
import yfinance as yf
import talib
import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

def technicalIndicators(df):

    #all Momentum indicators
    #awesome oscillator
    df.ta.ao(append=True)
    #Aboslute price oscillator
    df.ta.apo(append=True)
    #Bias
    df.ta.bias(append=True)
    #BRAR
    df.ta.brar(append=True)
    #commodity channel index
    df.ta.cci(append=True)
    #chande forecast oscillator
    df.ta.cfo(append=True)
    #center of gravity
    df.ta.cg(append=True)
    #Change momentum oscillator
    df.ta.cmo(append=True)
    #coppock curve
    df.ta.coppock(append=True)
    #Correlation trend indicator
    df.ta.cti(append=True)
    #Directional Movement
    df.ta.dm(append=True)
    #effeciency Ratio
    df.ta.er(append=True)
    #elder ray index
    df.ta.eri(append=True)
    #fisher trasnform
    df.ta.fisher(append=True)
    #inertia
    df.ta.inertia(append=True)
    #KDJ
    df.ta.kdj(append=True)
    #KST oscillator
    df.ta.kst(append=True)
    #Moving average convergence divergence
    df.ta.macd(append=True)
    #Momentum
    df.ta.mom(append=True)
    #Pretty good oscillator
    df.ta.pgo(append=True)
    #Percentage price Oscillator
    df.ta.ppo(append=True)
    #Psychological line
    df.ta.psl(append=True)
    #Percentage Volume Oscillator
    df.ta.pvo(append=True)
    #Rate of change
    df.ta.roc(append=True)
    #Relative strength index
    df.ta.rsi(append=True)
    #Relative vigor index
    df.ta.rvgi(append=True)
    #Schaff trend cycle
    df.ta.stc(append=True)
    #Slope
    df.ta.slope(append=True)
    #SMI Ergodic
    df.ta.smi(append=True)
    #Squeeze
    df.ta.squeeze(append=True)
    #Squeeze Pro
    df.ta.squeeze_pro(append=True)
    #Stochastic oscillator
    df.ta.stoch(append=True)
    #Stochastic RSI
    df.ta.stochrsi(append=True)
    #Ultimate Oscillator
    df.ta.uo(append=True)
    #Williams %R
    df.ta.willr(append=True)

    #Overlap technical indicators
    #Arnaud Legoux Moving average
    df.ta.alma(append=True)
    #Double Exponential Moving Average
    df.ta.dema(append=True)
    #Exponential Moving Average
    df.ta.ema(append=True)
    #Fibonacci's Weighted Moving Average
    df.ta.fwma(append=True)
    #High-Low Average
    df.ta.hl2(append=True)
    #High-Low-Close Average
    df.ta.hlc3(append=True)
    #Hull Exponential Moving Average
    df.ta.hma(append=True)
    #Holt-Winter Moving Average
    df.ta.hwma(append=True)
    #Jurik Moving Average
    df.ta.jma(append=True)
    #Kaufman's Adaptive Moving Average
    df.ta.kama(append=True)
    #Linear Regression
    df.ta.linreg(append=True)
    #McGinley Dynamic
    df.ta.mcgd(append=True)
    #Midpoint
    df.ta.midpoint(append=True)
    #Midprice
    df.ta.midprice(append=True)
    #Open-High-Low-Close Average
    df.ta.ohlc4(append=True)
    #Pascal's Weighted Moving Average
    df.ta.pwma(append=True)
    #WildeR's Moving Average
    df.ta.rma(append=True)
    #Sine Weighted Moving Average
    df.ta.sinwma(append=True)
    #Simple Moving Average
    df.ta.sma(append=True)
    #Ehler's Super Smoother Filter
    df.ta.ssf(append=True)
    #Supertrend
    # df.ta.supertrend(append=True)
    #Symmetric Weighted Moving Average
    df.ta.swma(append=True)
    #T3 Moving Average
    df.ta.t3(append=True)
    #Triple Exponential Moving Average
    df.ta.tema(append=True)
    #Triangular Moving Average
    df.ta.trima(append=True)
    #Variable Index Dynamic Average
    df.ta.vidya(append=True)
    #Volume Weighted Moving Average
    #df.ta.vwma(append=True)
    #Weighted Closing Price
    df.ta.wcp(append=True)
    #Weighted Moving Average
    df.ta.wma(append=True)
    #Zero Lag Moving Average
    df.ta.zlma(append=True)

    #Performance Technical indicators
    #Log Return
    df.ta.log_return(append=True)
    #Percent Return
    df.ta.percent_return(append=True)

    #Trend Technical Indicators
    #Average Directional Movement Index
    df.ta.adx(append=True)
    #Archer Moving Averages Trends
    df.ta.amat(append=True)
    #Aroon & Aroon Oscillator
    df.ta.aroon(append=True)
    #Choppiness Index
    df.ta.chop(append=True)
    #Chande Kroll Stop
    df.ta.cksp(append=True)
    #Decay
    df.ta.decay(append=True)
    #Decreasing
    df.ta.decreasing(append=True)
    #Increasing
    df.ta.increasing(append=True)
    #Long Run
    df.ta.long_run(append=True)
    #Q Stick
    df.ta.qstick(append=True)
    #Short Run
    df.ta.short_run(append=True)
    #Trend Signals
    df.ta.tsignals(append=True)
    #TTM Trend
    df.ta.ttm_trend(append=True)
    #Vertical Horizontal Filter
    df.ta.vhf(append=True)
    #Vortex
    df.ta.vortex(append=True)
    #Cross Signals
    df.ta.xsignals(append=True)

    #Volatility Technical Indicators
    #Aberration
    df.ta.aberration(append=True)
    #Acceleration Bands
    df.ta.accbands(append=True)
    #Average True Range
    df.ta.atr(append=True)
    #Bollinger Bands
    df.ta.bbands(append=True)
    #Donchian Channel
    df.ta.donchian(append=True)
    #Holt-Winter Channel
    df.ta.hwc(append=True)
    #Keltner Channel
    df.ta.kc(append=True)
    #Mass Index
    df.ta.massi(append=True)
    #Normalized Average True Range
    df.ta.natr(append=True)
    #Price Distance
    df.ta.pdist(append=True)
    #Relative Volatility Index
    df.ta.rvi(append=True)
    #Elder's Thermometer
    df.ta.thermo(append=True)
    #True Range
    df.ta.true_range(append=True)
    #Ulcer Index

    #Volume technical indicators
    df.ta.ui(append=True)
    #Accumulation/Distribution Index
    df.ta.ad(append=True)
    #Accumulation/Distribution Oscillator
    df.ta.adosc(append=True)
    #Archer On-Balance Volume
    df.ta.aobv(append=True)
    #Chaikin Money Flow
    #df.ta.cmf(append=True)
    #Elder's Force Index
    df.ta.efi(append=True)
    #Ease of Movement
    #df.ta.eom(append=True)
    #Klinger Volume Oscillator
    df.ta.kvo(append=True)
    #Money Flow Index
    df.ta.mfi(append=True)
    #Negative Volume Index
    df.ta.nvi(append=True)
    #On-Balance Volume
    df.ta.obv(append=True)
    #Positive Volume Index
    df.ta.pvi(append=True)
    #Price-Volume
    df.ta.pvol(append=True)
    #Price Volume Rank
    df.ta.pvr(append=True)
    #Price Volume Trend
    df.ta.pvt(append=True)


def transformDataFrame(df):
    #first drop the unnecessary rows
    print(df)
    df = df.iloc[70:]
    df.drop(df.tail(1).index,inplace=True) # drop last n rows

    #df = df.reset_index(drop=True)
    for i in range(68, len(df)-1):
        TI = []
        for item in df.iloc[i]:
            TI.append(item)
    print(df)
    return df
    




def calculateIncrease(df):
    dfTemp = df.iloc[70:]
    for i in range(0,len(dfTemp.Open)-1):
        print("voor ",dfTemp.iloc[i].Close, "na ", dfTemp.iloc[i+1].Close)
        difference = dfTemp.iloc[i+1].Close - dfTemp.iloc[i].Close
        percent = difference / dfTemp.iloc[i].Close * 100
        percentual_increase.append(percent)
        if percent > 0:
            increase.append(1)
        else:
            increase.append(0)
    return percentual_increase

def timeLoop(start_date, end_date, df):
    delta = dt.timedelta(hours=1)
    while start_date < end_date:
        print(start_date)

        
        start_date += delta

def machineLearning(df):
    df_train, df_test, increase_train, increase_test = train_test_split(df,increase, test_size = 0.5, random_state=69)
    svc_model = SVC()
    scaler = StandardScaler().fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    model = svc_model.fit(df_train, increase_train)
    increase_predict = model.predict(df_test)
    cm = np.array(confusion_matrix(increase_test, increase_predict, labels=[0,1]))
    confusion = pd.DataFrame(cm, index=['Increase', 'Decrease'], columns=['Predicted increase', 'Predicted Decrease'])
    print (confusion)
    exit()

def transformDate(date):
    date = date.split("-")
    datetimeFormat = dt.datetime(int(date[0]),int(date[1]),int(date[2]))
    dateFormat = dt.date(int(date[0]),int(date[1]),int(date[2]))
    return(datetimeFormat, dateFormat)

def main(start_date, end_date):
    #transforms start and end date
    start, start_date = transformDate(start_date)
    end, end_date = transformDate(end_date)
    #help(ta.yf)
    df = pd.DataFrame()
    #download data from 2 months and one week prior
    #we do this so that all TI's can be calculated
    #the +1 is sop that we can take the closing price of the day after end, this is for measwurement purposes
    df = yf.download('BTC-USD', start-dt.timedelta(days=3), end+dt.timedelta(days=1), interval="1h")
    technicalIndicators(df)
    #testing purposes TODO: remove
    #print(df)
    calculateIncrease(df)
    df = transformDataFrame(df)
    df.to_csv('nulltest.csv')
    exit()



    print(increase)
    print(df)
    timeLoop(start, end, df)
    machineLearning(df)
    delta = dt.timedelta(days=1)

    #df.ta.macd(cumulative=True)
    #pd.DataFrame('Close':[df.Close[]])
    #df.set_index(pd.DatetimeIndex(df["datetime"]), inplace=True)
    
    # coin = yf.download('BTC-USD', start+dt.timedelta(days=1), end)
    # #coin = coin.reset_index()
    # df = pd.DataFrame(data = coin, dtype= numpy.ndarray)
    # upperband, middleband, lowerband = talib.BBANDS(df.Close[1], timeperiod=1, nbdevup=2, nbdevdn=2, matype=0)

    # analysis = pd.DataFrame(index = df.index.values)

    # #dataframe = series.to_numpy()
    # timeLoop(start_date, end_date, df, analysis)


if __name__ == "__main__":
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    increase = []
    percentual_increase = []
    main(start_date, end_date)
