import sys
import io
import numpy as np
import zipfile
import hashlib
from pyspark import SparkContext, SparkConf
from tifffile import TiffFile
from scipy import linalg

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

def printresult(x,type):
    print("\n----", type, " Result----", x[0], x[1])
    return x

def intensifyPixel(x):
    m,n,r = x.shape
    intensityArr = np.zeros((m,n),dtype=int)
    for i in range(0,m):
        for j in range(0,n):

            #rgb_mean = int((x[i][j][0] + x[i][j][1] + x[i][j][2])/3)
            #intensity = rgb_mean * (x[i][j][3]/100)
            #intensity = int(((x[i][j][0] + x[i][j][1] + x[i][j][2])*x[i][j][3])/300)
            intensityArr[i][j] = int(sum(x[i][j][:-1])*x[i][j][3]/300)
            #intensityArr[i][j] = intensity
    return intensityArr

def meanIntensity(x,t):
    rowRange = range(0,x.shape[0],t)
    colRange = range(0,x.shape[1],t)
    reshaped = np.zeros((int(x.shape[0]/t),int(x.shape[1]/t)),dtype=int)
    iCtr = 0
    for row in rowRange:
        jCtr = 0
        for col in colRange:
            sum = 0
            #for i in range(0,t):
            #    for j in range(0,t):
            #        sum = sum + x[row+i][col+j]
            #mean = sum/(t*t)
            #reshaped[iCtr][jCtr] = mean
            reshaped[iCtr][jCtr] = np.mean(x[row:row+t, col:col+t])
            jCtr = jCtr + 1
        iCtr = iCtr + 1
    return reshaped

def rowdiff(x):
    row_diff = np.zeros((x.shape[0],x.shape[1]-1),dtype=int)
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]-1):
            row_ij = x[i][j+1]-x[i][j]
            if(row_ij > 1):
                row_ij = 1
            elif(row_ij < -1):
                row_ij = -1
            else:
                row_ij = 0
            row_diff[i][j] = row_ij
    y = np.reshape(row_diff, np.product(row_diff.shape))
    return y

def coldiff(x):
    col_diff = np.zeros((x.shape[0]-1,x.shape[1]),dtype=int)
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]-1):
            col_ij = x[j+1][i]-x[j][i]
            if(col_ij > 1):
                col_ij = 1
            elif(col_ij < -1):
                col_ij = -1
            else:
                col_ij = 0
            col_diff[j][i] = col_ij
    y = np.reshape(col_diff, np.product(col_diff.shape))
    return y

def sigImage(x,sign_size):
    length_x = len(x)
    chunk_size1 = length_x//sign_size
    chunk = []
    output = np.zeros((2),dtype=int)
    if(chunk_size1*sign_size != length_x):
        chunk_size2 = chunk_size1 + 1
        A = np.array([[chunk_size1, chunk_size2], [1, 1]])
        B = np.array([len(x), sign_size])
        output = np.linalg.solve(A,B)
        #first append lower size chunk
        ctr = 0
        for i in range(0,int(output[0])):
            chunk.append(x[ctr:ctr+chunk_size1])
            ctr = ctr + chunk_size1
        for i in range(0,sign_size - int(output[0])):
            chunk.append(x[ctr:ctr+chunk_size2])
            ctr = ctr + chunk_size2
    else:
        ctr = 0
        for i in range(0, sign_size):
            chunk.append(x[ctr:ctr+chunk_size1])
            ctr = ctr + chunk_size1

    #md5 hash function
    hash = hashlib.md5()
    hash_sign = []
    for i in range(0, sign_size):
        hash.update(str(chunk[i]).encode('utf-8'))
        hash_sign.append(int(hash.hexdigest(),16) % 10)
    return hash_sign
    #print(output.shape)

def dist_buckets(x,chunk_size,num_band,num_bucket):
    row_per_band = chunk_size//num_band
    bandCtr = 0
    result = np.zeros((num_band), dtype=int)
    for i in range(0, chunk_size, row_per_band):
        LSH_hash = int(''.join(map(str,x[i:i+row_per_band])))
        #LSH_hash = sum(x[i:i+row_per_band])
        LSH_hash_final = LSH_hash % num_bucket
        result[bandCtr] = LSH_hash_final
        bandCtr = bandCtr + 1
    return result


def simImg_search(x, img_toSearch):
    array = x[1]
    img_array = img_toSearch[0][1]
    for i in range(len(array)):
        if(array[i]==img_array[i]):
            return x[0]
    return None

def pca(iterator):
    y = []
    y1 = []
    for xi in iterator:
        y.append(np.array(xi[1]))
        y1.append(xi[0])
    img_diffs = np.asmatrix(np.array(y))
    #mu, std = np.mean(img_diffs, axis=0), np.std(img_diffs, axis=0)
    #img_diffs_zs = (img_diffs - mu) / std
    U, s, Vh = linalg.svd(img_diffs, full_matrices = 1)
    final = U[:,0:10]
    final_result = []
    for i in range(0,len(y1)):
        final_result.append([y1[i],final[i]])
    return final_result
    
def euclid_dist(similar_img, img_pca):
    for img in similar_img:
        ans = []
        for pca_i in img_pca:
            if(img[0] == pca_i[0]):
                img_pca_arr = pca_i[1]
        for img_sim in img[1]:
            for pca_i in img_pca:
                if(img_sim == pca_i[0]):
                    a = np.linalg.norm(pca_i[1]-img_pca_arr)
                    ans.append((img_sim,a))
        # sort on the basis of second values
        ans.sort(key=lambda x: x[1])
        # print for 1 then 18
        print("--- Step 3c Result ---", img[0], ans)

if __name__ == "__main__":
    #conf = SparkConf().setAppName("BDAssi2_vchanana").setMaster("local[2]")
    conf = SparkConf().setAppName("BDAssi2_vchanana")
    sc = SparkContext(conf=conf)

    #__VAISHALI__2017_11_15__Reading from files
    path_var = 'C:\\Vaishali Chanana\\SBU\\545 Big Data\\HW2\\a2_small_sample'
    #path_var = 'hdfs:/data/small_sample'
    # read path of zip folder from command line arguments
    #path_var = sys.argv[1]
    rdd = sc.binaryFiles(path_var)
    #rdd.persist()

    # 1(a)
    newRdd = rdd.map(lambda x: (x[0].split('/')[-1:][0], x[1]))

    smallResolution = 500
    # 1(b) ,1(c) and 1(d)
    tiffRdd = newRdd.map(lambda x: (x[0], getOrthoTif(x[1])))\
                    .map(lambda x:(x[0], reshapeTiff(x[1],smallResolution,"small")))\
                    .flatMap(lambda x: nameForImage(x))

    # for printing 1(e)
    oneRdd = tiffRdd.filter(lambda x: x[0] in {"3677454_2025195.zip-0","3677454_2025195.zip-1","3677454_2025195.zip-18","3677454_2025195.zip-19"})\
             .map(lambda x:(x[0],x[1][0][0])).map(lambda x: printresult(x,"Step 1")).collect()
    print("\n----------------Step 1 Results End----------------------\n")

    #__VAISHALI__2017_11_16__Part 2
    factor = 5
    newResolution = smallResolution//factor
    
    reduceRdd = tiffRdd.map(lambda x: (x[0], intensifyPixel(x[1])))\
                      .map(lambda x: (x[0], meanIntensity(x[1], factor)))

    #__VAISHALI__2017_11_17__Part 2 (c) and (d)
    rowDiffRdd = reduceRdd.map(lambda x: (x[0], rowdiff(x[1])))
    colDiffRdd = reduceRdd.map(lambda x: (x[0], coldiff(x[1])))
    featureRdd = rowDiffRdd.union(colDiffRdd).reduceByKey(lambda x,y: np.concatenate([x,y]))

    #for printing 2(f)
    twoRdd = featureRdd.filter(lambda x:x[0] in {"3677454_2025195.zip-1","3677454_2025195.zip-18"})\
                       .map(lambda x: printresult(x,"Step 2")).collect()
    print("\n----------------Step 2 Results End----------------------\n")
    #print(featureRdd.collect())

    #__VAISHALI__2017_11_21__Part 3
    chunk_size = 128
    num_bands = 2
    num_buckets = 12
    files = ["3677454_2025195.zip-0","3677454_2025195.zip-1","3677454_2025195.zip-18","3677454_2025195.zip-19"]
    threeRdd = featureRdd.map(lambda x: (x[0],sigImage(x[1],chunk_size)))\
                         .map(lambda x: (x[0],dist_buckets(x[1],chunk_size, num_bands, num_buckets))).persist()
    #print(threeRdd.collect())
    three_save = []
    for file in files:
        #print(file)
        fileRdd = threeRdd.filter(lambda x: x[0]==file).collect()
        threefinal = threeRdd.filter(lambda x: x[0]!= file).map(lambda x: simImg_search(x,fileRdd))\
                             .filter(lambda x: x is not None).collect()
        if(file in {"3677454_2025195.zip-1","3677454_2025195.zip-18"}):
            three_save.append((file,threefinal))
        print(len(threefinal))

    for i in three_save:
        printresult(i,"Step 3b")
    print("\n-----------------Step 3b Results End--------------------\n")

    #__VAISHALI__2017_11_25__Part 3c
    partition_num = 5
    pca_list = featureRdd.partitionBy(partition_num).mapPartitions(pca).collect()

    #Calculate Euclidean distance for the similar candidates
    euclid_dist(three_save,pca_list)
    print("\n-----------------Step 3c Results End--------------------\n")

