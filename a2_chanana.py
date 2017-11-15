import sys
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    conf = SparkConf().setAppName("BDAssi2_vchanana").setMaster("local[2]")
    sc = SparkContext(conf=conf)

    path_var = 'C:\\Vaishali Chanana\\SBU\\545 Big Data\\HW2\\a2_small_sample'
    rdd = sc.binaryFiles(path_var)

    print("Hi")