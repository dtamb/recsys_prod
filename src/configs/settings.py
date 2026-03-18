# settings.py
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, FloatType

# column titles
USER_COL = 'userId'
ITEM_COL = 'movieId'
TAG_ID_COL = 'tagId'
RATING_COL = 'rating'
TIMESTAMP_COL = 'timestamp'
RELEVANCE_COL = 'relevance'

# ratings schema structure
RATINGS_SCHEMA = StructType([
    StructField(USER_COL, IntegerType(), True),
    StructField(ITEM_COL, IntegerType(), True),
    StructField(RATING_COL, FloatType(), True),
    StructField(TIMESTAMP_COL, TimestampType(), True)
])

#  genomic schema structure
GENOMIC_SCHEMA = StructType([
    StructField(ITEM_COL, IntegerType(), True),
    StructField(TAG_ID_COL, IntegerType(), True),
    StructField(RELEVANCE_COL, FloatType(), True)
])