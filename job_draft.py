#!/usr/bin/env python3

from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, array, when, concat, sum, first, last
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import row_number,lit, when
from pyspark.sql.window import Window

from pyspark.sql.types import IntegerType, FloatType, StringType
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, DateType

spark = SparkSession     .builder     .appName("TCP data preparation")     .config("spark.some.config.option", "some-value")     .getOrCreate()

path = "data/sample/sample2.json"
rawDF = spark.read.json(path, multiLine=True)

def addColumn(df, alias, original):
    try:
        return df.withColumnRenamed(original, alias)
    except Exception as e:
        return df.withColumn(alias, lit(0))
    
df = rawDF.select('_source.layers.*').select('frame.*', 'ip.*', 'udp.*', 'ftp.*', 'tcp.*', 'tcp.`tcp.flags_tree`.*')

df = addColumn(df, 'start_time'   , 'frame.time')
df = addColumn(df, 'tcp_stream'   , 'tcp.stream')
df = addColumn(df, 'udp_stream'   , 'udp.stream')
df = addColumn(df, 'ip_src'       , 'ip.src')
df = addColumn(df, 'ip_dst'       , 'ip.dst')
df = addColumn(df, 'port_src'     , 'tcp.srcport')
df = addColumn(df, 'port_dst'     , 'tcp.dstport')
df = addColumn(df, 'ip_proto'     , 'ip.proto')
df = addColumn(df, 'frame_num'    , 'frame.number')
df = addColumn(df, 'data_len'     , 'frame.len')
df = addColumn(df, 'time_delta'   , 'frame.time_delta')
df = addColumn(df, 'time_relative', 'frame.time_relative')
df = addColumn(df, 'ip_flags'     , 'ip.flags')
df = addColumn(df, 'ftp_request'  , 'ftp.request')
df = addColumn(df, 'ftp_response' , 'ftp.response')
df = addColumn(df, 'urgent_bit'   , 'tcp.flags.urg')
df = addColumn(df, 'protocols'    , 'frame.protocols')
df = addColumn(df, 'serror_bit'   , lit(1))
df = addColumn(df, 'rerror_bit'   , lit(1))

df = df.withColumn('stream', F.udf(lambda x, y: x if y == None else y, StringType())('tcp_stream', 'udp_stream'))

df = df.select(*['start_time', 'ip_src', 'ip_dst', 'port_src', 'port_dst', 'ip_proto', 'frame_num', 'data_len', 
                'time_delta', 'time_relative', 'ip_flags', 'ftp_request', 'ftp_response', 'protocols', 'urgent_bit',
                'serror_bit', 'rerror_bit', 'stream'])

df = df.withColumn('ip_both', concat(
    when(df.ip_src > df.ip_dst, df.ip_src).otherwise(df.ip_dst),  # ip max
    when(df.ip_src > df.ip_dst, df.ip_dst).otherwise(df.ip_src))) # ip min
df = df.withColumn('unique', concat(col('ip_both'), col('ip_proto')))

df = df.na.drop(subset=['ip_both'])

df = df.fillna({'data_len': 0})

df = df.withColumn('data_len'     , col('data_len').cast(IntegerType()))
df = df.withColumn('time_delta'   , col('time_delta').cast(FloatType()))
df = df.withColumn('time_relative', col('time_relative').cast(FloatType()))
df = df.withColumn('ftp_request'  , col('ftp_request').cast(IntegerType()))

df = df.fillna({'protocols': ''})
df = df.withColumn('service', F.udf(lambda x: x.split(':')[-1], StringType())(df['protocols']))
df = df.withColumn('start_time', F.udf(lambda x: x.split(' ')[3].split('.')[0], StringType())(df['start_time']))
df = df.select([c for c in df.columns if c not in {'_index', '_type', '_source', 'protocols'}])

print("Dataframe step 1 done")

w = Window.partitionBy('stream').orderBy("frame_num")
df = df.withColumn('group_ip_src', first(df['ip_src']).over(w))
df = df.withColumn('group_ip_src', first(df['ip_src']).over(w))
df = df.withColumn('start_time', first(df['start_time']).over(w))

grouped = df.groupBy('stream').agg(
    F.first('start_time').alias('start_time'),
    F.first('ip_src').alias('ip_src'),
    F.first('ip_dst').alias('ip_dst'),
    F.first('port_src').alias('port_src'),
    F.first('port_dst').alias('port_dst'),
    F.first('ip_proto').alias('protocol'),
    F.first('service').alias('service'),
    F.last('time_relative').alias('time_relative'),
    F.sum('time_delta').alias('duration'),   
    F.sum(when(col('ip_src') == col('group_ip_src'), col('data_len'))).alias('src_bytes'),
    F.sum(when(col('ip_dst') == col('group_ip_src'), col('data_len'))).alias('dst_bytes'),

    F.count(when(col('ftp_request') == 1, col('ftp_request'))).alias('num_outbound_cmds'),
    F.count(when(col('urgent_bit') == '1', col('ftp_request'))).alias('urgent'),
    
    F.first('serror_bit').alias('serror'),
    F.first('rerror_bit').alias('rerror')
)

grouped = grouped.withColumn('land', col('ip_src') == col('ip_dst'))
grouped = grouped.fillna({'src_bytes': 0, 'dst_bytes': 0})

grouped = grouped.drop(*['diff', 'unique'])

"""
Calculating features count, srv_count, serror_rate, rerror_rate
-> Traffic features computed using a two-second time window.
"""
def count_same(ray, target):
    to_ret = 0
    for x in ray:
        to_ret += 1 if x == target else 0
    return to_ret

w = Window.orderBy('time_relative').rangeBetween(-2, 0)

grouped = grouped.withColumn("prev_conns_host", F.collect_list('ip_src').over(w))             .withColumn('count', F.udf(count_same, IntegerType())('prev_conns_host', 'ip_src'))             .drop('prev_conns_host')

grouped = grouped.withColumn("prev_conns_service", F.collect_list('service').over(w))             .withColumn('srv_count', F.udf(count_same, IntegerType())('prev_conns_service', 'service'))             .drop('prev_conns_service')

grouped = grouped.withColumn('error_count' , F.count('serror').over(w))             .withColumn('serror_count', F.count(col('serror') == '1').over(w))             .withColumn('rerror_count', F.count(col('rerror') == '1').over(w))             .withColumn('serror_rate', col('serror_count') / col('error_count'))             .withColumn('rerror_rate', col('rerror_count') / col('error_count'))             .drop(*['error_count', 'serror_count', 'rerror_count'])             .fillna({'serror_rate': 0, 'rerror_rate': 0})

print("Dataframe step 2 done")

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("start_date", StringType(), True),
    StructField("start_time", StringType(), True),
    StructField("duration", StringType(), True),
    StructField("service", StringType(), True),
    StructField("port_src", IntegerType(), True),
    StructField("port_dst", IntegerType(), True),
    StructField("ip_src", StringType(), True),
    StructField("ip_dst", StringType(), True),
    StructField("attack_score", StringType(), True),
    StructField("name", StringType(), True),
])

path = "data/sample/tcpdump.list"
labelDF = spark.read.csv(path, schema=schema, sep=' ')
labelDF.show()

resultDF = grouped.join(labelDF.select(*['start_time', 'ip_src', 'ip_dst', 'port_src', 'port_dst', 'attack_score']), 
                        ['start_time', 'ip_src', 'ip_dst', 'port_src', 'port_dst'],
                        "right_outer")


resultDF.write.csv('training.csv')
