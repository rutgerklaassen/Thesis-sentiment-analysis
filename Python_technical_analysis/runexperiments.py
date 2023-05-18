import os
import datetime as dt
from dateutil.relativedelta import relativedelta


if __name__ == "__main__":
    delta = dt.timedelta(days=1)
    start = dt.datetime(2022, 4, 1)
    end = dt.datetime(2022, 5, 1)

    actual_end = dt.datetime(2023, 2, 1)
    while end < actual_end: #this experiments script ONLY works with start and end dates that are on day 1 of the month
        plusOneEnd = end + relativedelta(months=1)
        if(start.month < 10):
            startString = str(start.year)+ "-0" +str(start.month)+ "-0" +str(start.day)
        else:
            startString = str(start.year)+ "-" +str(start.month)+ "-0" +str(start.day)
        if(end.month < 10):
            endString = str(end.year)+ "-0" +str(end.month)+ "-0" +str(end.day)
        else:
            endString = str(end.year)+ "-" +str(end.month)+ "-0" +str(end.day)
        if(plusOneEnd.month < 10 ):
            plusOneEndString = str(plusOneEnd.year)+ "-0" +str(plusOneEnd.month)+ "-0" +str(plusOneEnd.day)
        else:
            plusOneEndString = str(plusOneEnd.year)+ "-" +str(plusOneEnd.month)+ "-0" +str(plusOneEnd.day)
        modelString = startString + "-" + endString + "-Model"
        createModel = "python3 main.py " + startString + " " + endString + " S"
        os.system(createModel)

        testModel = "python3 main.py " + endString + " " + plusOneEndString + " " + modelString
        os.system(testModel)
        end = end + relativedelta(months=1)
        



