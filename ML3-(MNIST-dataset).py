# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:36:53 2018

@author: Shashwat
"""
##   Importing the necessary packages
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm

##  loading the data

#train_test-split breaks our data into two sets, 
#one for training and one for testing

labeled_images = pd.read_csv('mnist_data/train.csv')

#here we are taking the first 5000 examples
labels = labeled_images.iloc[0:5000,:1] #zeroeth column
images = labeled_images.iloc[0:5000,1:] #Rest of the columns

#splitting the data
train_images,test_images,train_labels,test_labels = train_test_split(images,labels, train_size=0.8, random_state=0)

# since the image is currently one-dimension
# we load it into a numpy array and reshape it so that it is 2D(28X28 px)
#Then we plot the image and label using matplotlib

i = 1   #plotting the first digit

#adding 1-D data for the digit
img = train_images.iloc[i].as_matrix()
#reshaping the data into 2D format
img = img.reshape((28,28))

#plotting the image of the digit
#print("Showing image in grayscale:")
plt.imshow(img,cmap="gray")
plt.title(train_labels.iloc[i,0])

##  Training our SVC model
#clf = svm.SVC()
#clf.fit(train_images, train_labels.values.ravel())
#score = clf.score(test_images, test_labels)
#print("SVC(for grayscale images) :%f"%(score))

#   Simplifying images to make them black and white
test_images[test_images>0] = 1
train_images[train_images>0] = 1

img = train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap="binary")
plt.title(train_labels.iloc[i])
#plt.show()

#plt.hist(train_images.iloc[i])
plt.hist(train_images.iloc[i])

clf = svm.SVC()
clf.fit(train_images,train_labels.values.ravel())
score = clf.score(test_images,test_labels)
print("SVC(for binary-scale images) :%f"%(score))

##  Labelling the test data

test_data = pd.read_csv("mnist_data/test.csv")
test_data[test_data>0] = 1
results = clf.predict(test_data[0:5000])

print(" The predicted result for the test.csv are: ")
print(results)

#plotting test.csv images

img = train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap="binary")
plt.title(train_labels.iloc[i])

df = pd.DataFrame(results)
df.index.name = "ImageId"
df.index+=1
df.columns = ["Label"]
df.to_csv("MNIST_results.csv",header=True)