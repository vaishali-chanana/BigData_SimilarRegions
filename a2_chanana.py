import sys
import io
import numpy as np
import zipfile
from pyspark import SparkContext, SparkConf
from tifffile import TiffFile

def getOrthoTif(zfBytes):
    #given a zipfile as bytes (i.e. from reading from a binary file),
    #  return a np array of rgbx values for each pixel
    bytesio = io.BytesIO(zfBytes)
    zfiles = zipfile.ZipFile(bytesio, "r")

    #find tif:
    for fn in zfiles.namelist():
        if fn[-4:] == '.tif':#found it, turn into array:
            tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
        return tif.asarray()

def reshapeTiff(x):
    m,n,r = x.shape
    t = 500
    rowRange = range(0,m,t)
    colRange = range(0,n,t)
    reshaped = []
    #iCtr = 0
    for row in rowRange:
        for col in colRange:
            #jCtr =0
            reshaped.append(x[row:row+t,col:col+t,:])
    return reshaped


def nameForImage(x):
    l = x[0]
    y = np.array(x[1])
    #print(y.shape)
    i,m,n,r = y.shape
    for cnt in range(i):
        v = l+"-"+str(cnt)
        yield(v,y[cnt])

def intensifyPixel(x):
    m,n,r = x.shape
    #intensityArr = np.zeros((m,n))
    intensityTest = []
    for i in range(0,m):
        for j in range(0,n):
            rgb_mean = (x[i][j][0] + x[i][j][1] + x[i][j][2])/3
            #intensity = int(rgb_mean * (x[i][j][3]/100))
            #intensityArr[i][j] = intensity
            intensityTest.append(rgb_mean)
    return intensityTest

if __name__ == "__main__":
    conf = SparkConf().setAppName("BDAssi2_vchanana").setMaster("local[2]")
    sc = SparkContext(conf=conf)

    #__VAISHALI__2017_11_15__Reading from files
    path_var = 'C:\\Vaishali Chanana\\SBU\\545 Big Data\\HW2\\a2_small_sample'
    # read path of zip folder from command line arguments
    #path var = sys.argv[1]
    rdd = sc.binaryFiles(path_var)
    #rdd.persist()

    # 1(a)
    newRdd = rdd.map(lambda x: (x[0].split('/')[-1:][0], x[1]))

    # 1(b) ,1(c) and 1(d)
    tiffRdd = newRdd.map(lambda x: (x[0], getOrthoTif(x[1]))).map(lambda x:(x[0], reshapeTiff(x[1]))).flatMap(lambda x: nameForImage(x))

    # for printing 1(e)
    oneRdd = tiffRdd.filter(lambda x: x[0] in {"3677454_2025190.zip-0","3677454_2025195.zip-1","3677454_2025195.zip-18","3677454_2025195.zip-19"}).map(lambda x:x[1][0][0])
    print("\n----------------Step 1 Results----------------------\n")
    #print(oneRdd.collect())

    #__VAISHALI__2017_11_16__Part 2
    featureRdd = tiffRdd.map(lambda x: (x[0], intensifyPixel(x[1])))
    print("\n----------------Step 2 Results----------------------\n")
    print(len(featureRdd.collect()))

