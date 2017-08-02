import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import training data from train.csv divide into x and y
def train_data():
	a= pd.read_csv("train.csv")
	data= np.array(a)
	y=data[:,0]
	x=data[:,1:]
	y=np.reshape(y,(42000,1))
	y_data=np.zeros((y.shape[0],10))
	for i in xrange(42000):
		ind=y[i]
		y_data[i][ind]=1
	return x,y_data

#import test data from test.csv
def test_data():
	b= pd.read_csv("test.csv")
	data= np.array(b)
	return data

def image(x):
	plt.imshow(x,cmap=plt.get_cmap('gray'))
	plt.show()
#show the grayscale pixels
def show(x):
	x=np.reshape(x,(28,28))
	plt.imshow(x,cmap=plt.get_cmap('gray'))
	plt.show()

#sigmoid function
def sigmoid(z):
	return 1/(1+np.e**(-z))

#derivative sigmoid function
def der_sigmoid(z):
	a=sigmoid(z)
	return a*(1-a)