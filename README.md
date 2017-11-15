# BigData_SimilarRegions

Overall Task: Find similar regions of Long Island by comparing satellite imagery.
Objectives:
1. Implement Locality Sensitive Hashing.
2. Implement dimensionality reduction.
3. Gain further experience with Spark.
4. Gain further experience with data preprocessing.
5. Explore a different modality of data: satellite images

Step 1. Read image files into an RDD and divide into 500x500 images (25 points)
Create an rdd of the orthographic zip files in the specified location. Pull the filenames out from the rest of the path (e.g. ‘/data/ortho/small_sample/3677453_2025190.zip’ => ‘3677453_2025190.zip’) and make sure to hang on to it.
Find all the tif files and convert them into arrays (make sure to hang on to the image filename).
Divide the image into 25 evenly sized subimages. Each tif image should be 2500x2500 pixels, so you will end up with 25 500x500 images (image breaking the image into a 5x5 grid).
Produce a final RDD where each record is of the form: (imagename, array).
**Print the r, g, b, x values for the pixel (0,0) in the following images::
3677454_2025190.zip-0, 3677454_2025195.zip-1, 3677454_2025195.zip-18, 3677454_2025195.zip-19

Step 2. Turn each image into a feature vector (25 points)
For each 500x500 image (each image should be an RDD record):
Convert the 4 values of each pixel into a single value, 
        intensity = int(rgb_mean * (infrared/100))
rgb_mean: the mean value of the red, green, and blue values for the pixel
infrared: (the 4th value in the tuple for the pixel
E.g. if the pixel was (10, 20, 30, 65), rgb_mean would be 20 and infrared would be 65. 
        Thus, intensity = int(20 * (65/100)) = 13
Reduce the resolution of each image by factor = 10. For example, from 500x500 to 50x50. To do this, take the mean intensity over factor x factor (10x10) sub-matrices. (Note one might build on the method for step 1(c); factor may change for later steps).
Compute the row difference in intensities. Direct intensity is not that useful because of shadows. Instead we focus on change in intensity by computing how it changes from one pixel to the next. Specifically we say intensity[i] = intensity[i+1] - intensity[i] (the numpy function ‘diff’ will do this). Convert all values < 1 to -1, those > 1 to 1, and the rest to 0. 
e.g. if the row was: [10, 5, 4, 10, 1], then the diff would be [-5, -1, 6, -9] and the vector after conversion would be [11, 0, 1, -1] (notice there is one less column after doing this). We call this row_diff
Compute the column difference in intensities. Do this over the direct intensity scores -- not row_diff. We call this col_diff
Turn row_diff and col_diff into one long feature vector. Row_diff and col_diff are matrices. Flatten them into long vectors, then append them together. Cal this features. Row_diff has 49x50 = 2450 values and col_diff hass 50x49 = 2450 values. Thus, features will have 2450 + 2450 = 4900 values,
**Print the feature vectors for: 3677454_2025195.zip-1, 3677454_2025195.zip-18,*

Step 3. Use LSH, PCA to find similar images (50 points)
Create a 16 byte “signature” for each image by passing the feature vector to a 16-byte md5 hash. (use digest to represent the 16 bytes)
Out of all images, run LSH to find approximately 20 candidates that are most similar to images: 3677454_2025195.zip-0, 3677454_2025195.zip-1, 3677454_2025195.zip-18, 3677454_2025195.zip-19
Tweak brands and rows per band in order to get approximately 20 candidates (i.e. anything between 15 to 25 candidates is ok)
Note that in LSH the columns are signatures. Here each RDD record is a signature, and so from the perspective of LSH, each record is a column.
While pre existing hash code is fine, you must implement LSH otherwise. 
**print the 20 candidates for 3677454_2025195.zip-1, 3677454_2025195.zip-1
For each of the candidates from (b), use PCA and find the euclidean difference in low dimensional feature space of the images. 
Start by running PCA across the 4900 dimensional feature vectors of *all* images to reduce the feature vectors to only 10 dimensions. Use the random batched SVD approach.
Then (can be done outside RDDs), compute the euclidean distance between 3677454_2025195.zip-1, 3677454_2025195.zip-18, and each of their candidates within this 10 dimensional space.
**print the distance scores along with their associated imagenames, sorted from least to greatest
