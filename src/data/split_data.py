# split_data.py

### Chronological and User based training and test split ###
#
#  Splits up training and test so that each user is represented in both
# training and test data. Also maintains chronological sequence so that all
# training datapoints are prior to test ones for each given user
from pyspark.sql import Window
from pyspark.sql import functions as F

def chron_user_tt_split (df, userCol, timestampCol, threshold):
    w = Window.partitionBy(userCol).orderBy(timestampCol)
    df_rank = df.withColumn("pr", F.percent_rank().over(w))
    
    train = df_rank.filter(df_rank.pr <= threshold)
    test = df_rank.filter(df_rank.pr > threshold)
    return train, test

# NB: percent_rank is an approximation for the train-test split as some users submitted multiple ratings at the same timestamp
# For MovieLens20, this worked out to 0.802 and was deemed acceptable