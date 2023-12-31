# Face-Recognition-Classification Algorithms

## Table of Contents

1. [Face Recognition: k-NN](#face-recognition-k-nn)
2. [Eigenfaces (PCA)](#eigenfaces-pca)
3. [Fisherfaces (LDA)](#fisherfaces-lda)
4. [Support Vector Machine](#support-vector-machine)
5. [Sparse-Representation Classification](#sparse-representation-classification)


## Extended YaleB Dataset Analysis

### Dataset Overview:
The **Extended YaleB dataset** comprises images of 38 unique individuals, each having approximately 64 near-frontal photos under varying illumination conditions. The original images have been processed to a resolution of $32 \times 32$ pixels. 
### Dataset File:
The dataset can be downloaded as `YaleB-32x32.mat` from the course locker. It contains two main variables:
- `fea`: Each row of this variable represents a face.
- `gnd`: This variable provides the label for each face in `fea`.

### Experiment Details:
For the purpose of our experiment, we randomly select $(m=10,20,30,40,50)$ images per individual to serve as the training dataset, and the remaining images are used as the test set. We employ the $k$-NN algorithm for classification, using the Euclidean distance metric:

$$
d(x, y)=\|x-y\|_2
$$

The classification error rate, $E$, is given by:

$$
E=\frac{\sum_{i=1}^n \mathbb{1}\left[\hat{l}\left(x_i\right) \neq l\left(x_i\right)\right]}{n} \times 100
$$

Where:
- $n$ is the total number of test samples.
- $\hat{l}\left(x_i\right)$ represents the classified label for the $i$-th observation from the test set.
- $l\left(x_i\right)$ is the actual label for the $i$-th observation.


## Face Recognition: k-NN

Please see **`knn.ipynb`**

### 1. For k=1, Plot the Classification Error Rate vs Number of Trainings Samples

$k=1$

| m images | Classification Error Rate |
| :---: | :---: |
| 10 | 56.04719764011799 |
| 20 | 40.87061668681983 |
| 30 | 34.69387755102041 |
| 40 | 28.299776286353467 |
| 50 | 24.124513618677042 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/4473500a-7fb0-45ba-98c2-92dca27aa401)

This code will loop over different numbers of training samples (10, 20, 30, 40, 50), randomly select m
images per individual for the training set, and use the remaining images as the test set. For each split, it
will apply the k-NN algorithm with k=1 and the Euclidean distance metric, and calculate the classification
error rate. The final output will be a plot of the classification error rate vs the number of training samples.
Based on the plot, we can see that the classification error rate decreases as the number of training samples
increases. This is expected, as having more training data allows the model to better capture the patterns
in the data and make more accurate predictions. However, the rate of improvement starts to slow down as
the number of training samples gets larger. In this case, using around 30 training samples per individual
seems to give good performance without requiring too much training data.

### 2. For k = 2, 3, 5, 10, error rate E against k and misclassified samples

- $k=2$

| $m$ | Classification Error Rate |
| :---: | :---: |
| 10 | 64.30678466076697 |
| 20 | 51.511487303506655 |
| 30 | 42.700156985871274 |
| 40 | 39.14988814317674 |
| 50 | 32.87937743190661 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/31839f33-270f-42ae-a1da-58f62ce9ca48)

<div align="center">

*Misclassified sample for k = 2*

</div>

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/f9b29479-5051-4ef6-b76e-eeedefd88386)


<br>


- $k=3$

| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 68.68239921337266 |
| 20 | 53.3857315598549 |
| 30 | 45.368916797488225 |
| 40 | 39.038031319910516 |
| 50 | 35.992217898832685 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/d7f2cf87-6933-4c62-b5e7-5d3b204c94cd)

<div align="center">

*Misclassified sample for k = 3*

</div>

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/8a68c580-bc88-4706-94e6-b31efaf13bbd)

<br>


- $k=5$
  
| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 72.02556538839725 |
| 20 | 55.139056831922616 |
| 30 | 44.81946624803768 |
| 40 | 40.380313199105146 |
| 50 | 32.68482490272373 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/a25e484c-531f-4f6c-9579-db65a78e8209)

<div align="center">

*Misclassified sample for k = 5*

</div>

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/73a6d607-95f7-4c0a-9b20-c17eb98a8549)


<br>


- $k=10$
  
| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 75.51622418879056 |
| 20 | 59.008464328899635 |
| 30 | 51.25588697017268 |
| 40 | 42.281879194630875 |
| 50 | 37.7431906614786 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/364fd402-75b0-4d5e-a9d4-cb9cc894286a)


<div align="center">

*Misclassified sample for k = 10*

</div>

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/020d44aa-fe4a-444b-88a7-2f261ceadf04)


<br>


calculated the average classification error rate for each k-value across all
m-values. It shows the average classification error rate did increase instead of getting lower, which is what
I expected. Calculate average classification error rate across splits for current value of k.


- *k-neighbor’s average classification error rate across m*

| $\mathbf{k}$ | Average Classification Error Rate |
| :---: | :---: |
| 1 | 36.85898966795061 |
| 2 | 37.054154233255616 |
| 3 | 37.54995281372114 |
| 5 | 37.317008582744066 |
| 10 | 37.43568694370498 |


![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/09edd929-3dfc-4530-a1a5-913561534f96)


The problem specifically asks to plot the error rate E against k. The best way to showcase this is to
display the increasing classification error rates for each choice of m images used for the training set.


- *k-neighbor’s classification error rate for m = 10*
  
| $\mathbf{k}$ | Classification Error Rate |
| :---: | :---: |
| 1 | 56.04719764011799 |
| 2 | 64.30678466076697 |
| 3 | 68.68239921337266 |
| 5 | 72.02556538839725 |
| 10 | 75.51622418879056 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/9a5eb3f3-2137-4c23-a18f-fb6b14f5e8b6)


- *k-neighbor’s classification error rate for m = 20*

| $\mathbf{k}$ | Classification Error Rate |
| :---: | :---: |
| 1 | 40.87061668681983 |
| 2 | 51.511487303506655 |
| 3 | 53.3857315598549 |
| 5 | 55.139056831922616 |
| 10 | 59.008464328899635 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/9d458713-0e71-427a-a6c8-23356032a61c)



- *k-neighbor’s classification error rate for m = 30*
  
| $\mathbf{k}$ | Classification Error Rate |
| :---: | :---: |
| 1 | 34.69387755102041 |
| 2 | 42.700156985871274 |
| 3 | 45.368916797488225 |
| 5 | 44.81946624803768 |
| 10 | 51.25588697017268 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/6330d185-717a-4fb0-b606-128048647c9d)



- *k-neighbor’s classification error rate for m = 40*

| $\mathbf{k}$ | Classification Error Rate |
| :---: | :---: |
| 1 | 28.299776286353467 |
| 2 | 39.14988814317674 |
| 3 | 39.038031319910516 |
| 5 | 40.380313199105146 |
| 10 | 42.281879194630875 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/dc70e268-1d4e-405f-8ba6-335ba65bd595)


- *k-neighbor’s classification error rate for m = 50*
  
| $\mathbf{k}$ | Classification Error Rate |
| :---: | :---: |
| 1 | 24.124513618677042 |
| 2 | 32.87937743190661 |
| 3 | 35.992217898832685 |
| 5 | 32.68482490272373 |
| 10 | 37.7431906614786 |
  
![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/1dd336ad-cdea-41d5-84c9-930477b44232)


The error rate increases with larger values of $k$, contrary to the expectation of improved accuracy. This could be due to the noisy raw pixel data, making generalization challenging. More neighbors might amplify the noise's impact. Despite the typical advantage of adding more training samples, the inconsistent quality of these samples might adversely affect performance. The trend of rising error rates is observed even with just two neighbors, suggesting that overfitting might not be the primary issue. 

### 3. Let k = 3 and select m = 30 images per individual with labels to form the training set and use the remaining images in the dataset as the test set. $\|x-y\|_p$
- *Replace Distance Metric (error rate vs p )*

| $\mathbf{p}$ | Classification Error Rate |
| :---: | :---: |
| 1 | 47.48822605965463 |
| 3 | 40.816326530612244 |
| 5 | 39.403453689167975 |
| 10 | 46.23233908948195 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/951f2753-1ca3-4d78-98cd-72446622b126)


Based on the plot, we can see that the choice of distance metric does affect the classification error rate. In
this case, the classification error rate generally decreases as the value of p increases, except for an increase
when p=10. P = 5 has the lowest error rate in this test. This is because a larger p value means the distance metric places more
emphasis on the larger differences between feature values, which can make the classification more robust to
outliers and noise. However, a very large p value can also lead to overfitting and decreased generalization
performance. 


### 4. Instead of using the pixel intensities as features, extract the LBP and HOG features from the images. Repeat step 3 with p = 1, 2. What are the error rates corresponding to pixel intensities, LBP and HOG features?

- *Compare error rates for LBP and HOG features (along with pixel intensities as features)*

|  | $\mathrm{P}=1$ | $\mathrm{P}=2$ |
| :--- | :--- | :--- |
| Pixels | 51.41287284144427 | 45.29042386185244 |
| LBP | 0.9419152276295133 | 0.7064364207221351 |
| HOG | 57.53532182103611 | 57.927786499215074 |

from skimage.feature import hog, local binary pattern To extract the LBP and HOG features from the
images and apply k-NN classification with different distance metrics,

### 5. Lowest Error Rate
The lowest error rate is actually achieved using LBP as the main feature. Unless my implementation is wrong, the classification errors were 0.94% and .71% (both extremely low).

### 6. Validation Set
Created a script to optimize the parameters of a k-Nearest Neighbors (k-NN) algorithm using a validation
set. It loops through different values for the parameters k and p, and finds the combination that gives the
lowest classification score on the validation set.
For each combination of k and p values, the algorithm is trained on the training data and labels, and then
it is used to predict the validation set labels. The accuracy of the predictions is calculated, and if it is
lower than the lowest validation error seen so far, the lowest validation error and corresponding k and p
values are updated.  
- Test Error Rate: 28.68%  
- Best k: 1  
- Best p: 3  

As an extra step to make sure the parameter updated were done correctly, I compared the best parameter
results from my script to the results outputted from sklearn’s knn.score. Then, it trains the k-NN algorithm on the new training set using those optimized parameters, and evaluates its performance on the test set. Finally, it prints out the accuracy of the model on the test set and best parameters found.
- Test Accuracy: 0.753
- neighbors: 1
- p: 3



## Eigenfaces (PCA)

Please see **`eigenfaces.ipynb`**

For PCA/Eigenfaces, I used both built-in functions from sklearn and also attempted to build my implementation of PCA/eigenfaces from scratch. PCA was applied to the training data and transformed both the training and test data using the PCA model. A k-NN classifier was trained on the transformed training data and predicted labels for the test data.

### 1. sklearn's built-in PCA function 

Using sklearn’s built-in function ``PCA(n_components=100)``, 

- *PCA Components = 100, For k = 1*
  
| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 59.63618485742379 |
| 20 | 46.43288996372431 |
| 30 | 39.71742543171114 |
| 40 | 34.22818791946309 |
| 50 | 30.35019455252918 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/5a3b3a9b-514e-4908-ba82-ee7c97eed65b)

These classification error rates are comparable to the kNN results without PCA applied (the error rates
are slightly worse with PCA).

- *For k = 1*

| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 56.04719764011799 |
| 20 | 40.87061668681983 |
| 30 | 34.69387755102041 |
| 40 | 28.299776286353467 |
| 50 | 24.124513618677042 |


### 2. sklearn's built-in PCA functions with svd and whiten parameters

``PCA(n_components=100, svd_solver='randomized', whiten=True)``
Using the Principal Component Analysis (PCA) function with the following parameters: n components=100,
svd solver=’randomized’, and whiten=True produced even lower classification rates compared to regular
PCA. The PCA object will be used to reduce the dimensionality of a dataset by projecting it onto a lowerdimensional space, while preserving as much of the variance as possible. The n components parameter specifies the number of components to keep in the reduced dataset (in this case, 100). The svd solver
parameter specifies which algorithm to use for computing the Singular Value Decomposition (SVD) of the
data; in this case, ’randomized’ is specified, which uses a randomized algorithm for faster computation.
The whiten parameter specifies whether or not to normalize each component so that it has unit variance;
in this case, it is set to True.

- *PCA - SVD, whiten Error Rates*
- 
| $\mathbf{m}$ | AverageClassificationErrorRate |
| :---: | :---: |
| 10 | 23.35299901671583 |
| 20 | 15.235792019347038 |
| 30 | 11.852433281004709 |
| 40 | 10.40268456375839 |
| 50 | $9.33852140077821]$ |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/07f4da46-ee8b-4c68-bf67-e9cd7f1b8ad0)

### 3. PCA/Eigenfaces implemented from scratch

implemented the eigenfaces algorithm for facial recognition from scratch. It is calculating the average
face from the training data, normalizing both the training and test faces, calculating the covariance matrix and
eigenvalues/eigenvectors, sorting the eigenvectors by decreasing eigenvalues, selecting a number of eigenfaces to
use (K), projecting normalized training faces onto K eigenvectors, predicting labels for test set, and calculating
the classification error rate. The classification errors of my eigenfaces implementation was comparable to the
errors from sklearn’s built in PCA function.

- *Eigenfaces*

| $\mathbf{m}$ | ClassificationErrorRate |
| :---: | :---: |
| 10 | 59.24287118977385 |
| 20 | 47.40024183796856 |
| 30 | 41.679748822605966 |
| 40 | 37.13646532438479 |
| 50 | 34.04669260700389 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/bb6838d1-cbd2-4292-b9ca-3192087e8952)


## Fisherfaces (LDA)

Please see **`fisherfaces.ipynb`**

For LDA/Fisherfaces, I used both built-in functions from sklearn and attempted to build an implementation from scratch. Performing linear discriminant analysis (LDA) on the training data and test data. It then uses the LDA-transformed data to train a k-nearest neighbors (k-NN) algorithm. The code then predicts labels for the test set using the k-NN algorithm and calculates the classification error rate.

### 1. Built-in LDA

Using sklearn’s built-in function ``LinearDiscriminantAnalysis(n_components=min(37, len(np.unique(train_labels))))``

- *LDA*

| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 22.566371681415927 |
| 20 | 14.87303506650 |
| 30 | 20.015698587127 |
| 40 | 4.47427293 |
| 50 | 3.307392996108 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/87edc108-c578-40ca-ad92-451f3ec94f6e)

LDA has better accuracy than PCA because it is a supervised learning algorithm, meaning it uses labeled
data to make predictions. PCA is an unsupervised learning algorithm, meaning it does not use labeled data
to make predictions. LDA also takes into account the underlying structure of the data, which can improve its
accuracy. Additionally, LDA can be used for classification tasks, whereas PCA is mainly used for dimensionality
reduction

### 2. LDA from scratch

Also implemented the LDA fisherfaces from scratch.This code implements the Linear Discriminant Analysis
(LDA) Fisherfaces algorithm. It takes in training data, training labels, testing data, testing labels, and the
number of components as input. It then computes the mean face from the training data and computes the
difference faces by subtracting the mean face from each of the training data points. It then computes both the
within-class scatter matrix Sw and between-class scatter matrix Sb. Next, it solves a generalized eigenvalue
problem to compute an LDA matrix and projects both the train and test data onto this LDA subspace. Finally, it predicts test labels by finding the closest label in the training data for each test point and calculates a classification error rate based on how many of these predictions are correct.

- *LDA from scratch*

| $\mathbf{m}$ | ClassificationError Rate |
| :---: | :---: |
| 10 | 98.13176007866274 |
| 20 | 97.03748488512697 |
| 30 | 19.23076923076923 |
| 40 | 5.9284116331096195 |
| 50 | $3.307392996108949]$ |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/6445df83-cc3b-4a13-9fbd-39b45a3dee13)

I am not sure why the classification error rate is so high for m=10 and m=20, but the classification errors
for m=30, 40, and 50 remain comparable to the built-in LDA function classification error. I also implemented
a function that computes the Linear Discriminant Analysis (LDA) projection matrix for a given training data
set and labels. It first computes the class-wise mean vectors, then calculates the within-class scatter matrix and
between-class scatter matrix. It then calculates the eigenvectors and eigenvalues of $(Sw−1).Sb$, sorts them in
descending order of eigenvalues, and finally returns the LDA projection matrix.

---

## Support Vector Machine

Please see **`svm.ipynb`**

- *Support Vector Machine using sklearn’s built-in function.*

| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 28.56440511307768 |
| 20 | 12.515114873035065 |
| 30 | 8.712715855572998 |
| 40 | 8.165548098434003 |
| 50 | $7.003891050583658]$ |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/bf5d68b0-20da-4842-82d0-d61bdad7daf0)

The results for classification
error rate are comparable to LDA because both are supervised linear classifiers. Support Vector Machine’s
classification error was slightly higher.


## Sparse-Representation Classification

Please see **`src_matchingpursuit.ipynb`**

Orthogonal Matching Pursuit (OMP) is the sparsity-promoting algorithm used in the dataset. The matching
pursuit model is trained on each test sample individually using the entire dictionary. This implementation first
extracts patches from each test image, and then uses the trained dictionary and Orthogonal Matching Pursuit to obtain the sparse representation coefficients for each patch. The coefficients for each test image are then
averaged to obtain a single coefficient vector for the image, which is used to predict the label. Unless my
algorithm was implemented incorrectly, the results from this have extremely low classification errors.

| $\mathbf{m}$ | Classification Error Rate |
| :---: | :---: |
| 10 | 0.04916420845624385607768 |
| 20 | 0.06045949214026602 |
| 30 | 0.07849293563579278 |
| 40 | 0.11185682326621924 |
| 50 | 0.19455252918287938 |

![image](https://github.com/travislatchman/Face-Recognition-Classification-Algorithms/assets/32372013/fac0e3ca-65ef-45a6-bdad-27a59f4214d0)


