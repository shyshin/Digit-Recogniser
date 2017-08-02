import data
import numpy as np
import matplotlib.pyplot as plt
from dat import sigmoid,der_sigmoid,train_data,test_data
from scipy.optimize import minimize
from IPython.display import clear_output
class Network:
	
	def __init__(self,l,size=0):
		self.m=l.shape[0]
		self.input=l.shape[1]
		#theta initialisations
		self.theta1=np.random.randn(self.input,15)
		self.theta2=np.random.randn(16,15)
		self.theta3=np.random.randn(16,10)
		#activation 
		self.activate1=l
		self.activate4=np.random.randn(self.m,10)
		self.activate2=np.random.randn(self.m,16)
		self.activate3=np.random.randn(self.m,16)
		#partial derivative wrt to theta
		self.del4=np.random.randn(self.m,10)
		self.del2=np.random.randn(self.m,16)
		self.del3=np.random.randn(self.m,16)
		
	def pr(self):
		return self.activate2.shape,self.activate3.shape,self.activate4.shape
	
	def feedforward(self,l):
		a=l
		a=a.dot(self.theta1)
		b=np.ones((a.shape[0],16))
		b[:,1:]=sigmoid(a)
		a=b
		self.activate2=a
		a=a.dot(self.theta2)
		b=np.ones((a.shape[0],16))
		b[:,1:]=sigmoid(a)
		a=b
		self.activate3=a
		a=a.dot(self.theta3)
		a=sigmoid(a)
		self.activate4=a
		return a
	
	def backprop(self,y):
		self.del4=self.activate4-y
		self.del3=(self.del4.dot((self.theta3).T))*(self.activate3*(1-self.activate3))
		self.del2=(self.del3[:,1:].dot(self.theta2.T))*(self.activate2*(1-self.activate2))
		return self.del4,self.del3,self.del2
	
	def cost(self,lam,y):
		J=-y*(np.log(sigmoid(self.activate4)))-(1-y)*(np.log(1-sigmoid(self.activate4)))
		J=np.sum(J)
		J+=(lam/2)*(np.sum(self.theta1*self.theta1)+np.sum(self.theta3[:,1:]*self.theta3[:,1:])+np.sum(self.theta2[:,1:]*self.theta2[:,1:]))
		J/=self.m
		return J
		
	def der(self,lam):
		der4=0
		der4+= self.activate3.T.dot(self.del4)
		der3=self.activate2.T.dot(self.del3[:,1:])
		der2=self.activate1.T.dot(self.del2[:,1:])
		
		der4/=self.m
		der3/=self.m
		der2/=self.m
		
		der4[:,1:]+=lam*(self.theta3[:,1:])
		grad4=np.reshape(der4,(der4.shape[0]*der4.shape[1],1))
		
		der3[:,1:]+=lam*(self.theta2[:,1:])
		grad3=np.reshape(der3,(der3.shape[0]*der3.shape[1],1))
		
		der2[:,1:]+=lam*(self.theta1[:,1:])
		grad2=np.reshape(der2,(der2.shape[0]*der2.shape[1],1))
		
		grad=np.concatenate((grad2,grad3,grad4))
		grad=np.reshape(grad,(grad.shape[0]*grad.shape[1],1))
		return der2,der3,der4
	
	def grad_descent(self,alpha,it):
		x,y= train_data()
		a=self.feedforward(x)
		b=self.backprop(y)
		g1,g2,g3=self.der(0.1)
		for j in xrange(it):
			print("iteration: "+str(j)+"|training:")
			plt.imshow(self.theta2,cmap=plt.get_cmap('gray'))
			plt.show()
			t1=self.theta1-(alpha*g1)
			t2=self.theta2-(alpha*g2)
			t3=self.theta3-(alpha*g3)
			self.theta1=t1
			self.theta2=t2
			self.theta3=t3
			a=self.feedforward(x)
			b=self.backprop(y)
			g1,g2,g3=self.der(0.1)
	def grad(self,epsilon,lam,y):
		
		for i in xrange(self.theta1.shape[0]):
			for j in xrange(self.theta1.shape[1]):
				t1=self.theta1
				self.theta1[i][j]+=epsilon
				c_plus=self.cost(lam,y)
				self.theta1=t1
				self.theta1[i][j]-=epsilon
				c_minus=self.cost(lam,y)
				


				
	def output(self):
		x=test_data()
		y=self.feedforward(x)
		y_label=np.argmax(y,axis=1)
		return y_label