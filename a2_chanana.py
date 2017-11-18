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

def reshapeTiff(x,t,type):
    #m,n,r = x.shape
    #t = 500
    rowRange = range(0,x.shape[0],t)
    colRange = range(0,x.shape[1],t)
    reshaped = []
    #iCtr = 0
    for row in rowRange:
        for col in colRange:
            #jCtr =0
            if(type=="small"):
                reshaped.append(x[row:row+t,col:col+t,:])
            #elif(type=="new"):
            #    reshaped.append(x[row:row+t,col:col+t])
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
    intensityArr = np.zeros((m,n))
    #intensityTest = []
    for i in range(0,m):
        for j in range(0,n):
            rgb_mean = (x[i][j][0] + x[i][j][1] + x[i][j][2])/3
            intensity = int(rgb_mean * (x[i][j][3]/100))
            intensityArr[i][j] = intensity
            #intensityTest.append(rgb_mean)
    return intensityArr

def meanIntensity(x,t):
    rowRange = range(0,x.shape[0],t)
    colRange = range(0,x.shape[1],t)
    reshaped = np.zeros((int(x.shape[0]/t),int(x.shape[1]/t)))
    iCtr = 0
    for row in rowRange:
        jCtr = 0
        for col in colRange:
            sum = 0
            for i in range(0,t):
                for j in range(0,t):
                    sum+=x[row+i][col+j]
            mean = sum/(t*t)
            reshaped[iCtr][jCtr] = mean
            jCtr = jCtr + 1
        iCtr = iCtr + 1
    return reshaped

def rowdiff(x):
    row_diff = np.zeros((x.shape[0],x.shape[1]-1))
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]-1):
            row_ij = x[i][j+1]-x[i][j]
            if(row_ij > 1):
                row_ij = 1
            elif(row_ij < 1):
                row_ij = -1
            else:
                row_ij = 0
            row_diff[i][j] = row_ij
    y = np.reshape(row_diff, np.product(row_diff.shape))
    return y

def coldiff(x):
    col_diff = np.zeros((x.shape[0]-1,x.shape[1]))
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]-1):
            col_ij = x[j+1][i]-x[j][i]
            if(col_ij > 1):
                col_ij = 1
            elif(col_ij < 1):
                col_ij = -1
            else:
                col_ij = 0
            col_diff[j][i] = col_ij
    y = np.reshape(col_diff, np.product(col_diff.shape))
    return y

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

    smallResolution = 500
    # 1(b) ,1(c) and 1(d)
    tiffRdd = newRdd.map(lambda x: (x[0], getOrthoTif(x[1]))).map(lambda x:(x[0], reshapeTiff(x[1],smallResolution,"small"))).flatMap(lambda x: nameForImage(x))

    # for printing 1(e)
    oneRdd = tiffRdd.filter(lambda x: x[0] in {"3677454_2025190.zip-0","3677454_2025195.zip-1","3677454_2025195.zip-18","3677454_2025195.zip-19"}).map(lambda x:x[1][0][0])
    print("\n----------------Step 1 Results----------------------\n")
    #print(oneRdd.collect())

    #__VAISHALI__2017_11_16__Part 2
    factor = 10
    #newResolution = int(smallResolution/factor)
    reduceRdd = tiffRdd.map(lambda x: (x[0], intensifyPixel(x[1]))).map(lambda x: (x[0], meanIntensity(x[1], factor)))

    #__VAISHALI__2017_11_17__Part 2 (c) and (d)
    rowDiffRdd = reduceRdd.map(lambda x: (x[0], rowdiff(x[1])))
    colDiffRdd = reduceRdd.map(lambda x: (x[0], coldiff(x[1])))
    featureRdd = rowDiffRdd.union(colDiffRdd).reduceByKey(lambda x,y: np.concatenate([x,y]))

    #for printing 2(f)
    twoRdd = featureRdd.filter(lambda x:x[0] in {"3677454_2025195.zip-1","3677454_2025195.zip-18"})
    print("\n----------------Step 2 Results----------------------\n")
    #print(twoRdd.collect())

