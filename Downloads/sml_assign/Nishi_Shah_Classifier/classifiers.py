from __future__ import division
import pandas
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class bayesClassify:
 def __init__(self,file1,frac):
  self.dataset=np.genfromtxt(file1,delimiter=",")
  self.dataset=sc.delete(self.dataset,0,1) #discarding first column which is the id number
  for i in range(self.dataset.shape[0]): #removig the data entries with null value
   if (np.isnan(self.dataset[i,5])):
    np.delete(self.dataset,i,0)
  np.random.shuffle(self.dataset) # randomize dataset for better accuracy
  for x in range(self.dataset.shape[0]):
   if(self.dataset[x,9]==2):
    self.dataset[x,9]=1
   elif(self.dataset[x,9]==4):
    self.dataset[x,9]=-1
  frac=int((frac*self.dataset.shape[0]*2)/(100*3)) # taking only fraction part of 2/3 training set for training
  self.train, self.test = self.dataset[:frac,:], self.dataset[467:,:] # Partitioning the dataset into training and test data for cross validation
  self.yes=0
  self.no=0
  self.total=0
  

 def likelihood_pos(self,index,feature):
  self.total=0
  self.yes=0
  feature_yes=0
  for i in range(0,self.train.shape[0]):
   self.total=self.total+1
   if (self.train[i,9]==1):
    self.yes=self.yes+1
    if(self.train[i,index]==feature):
      feature_yes=feature_yes+1
  return ((feature_yes+1)/(self.yes+9)) #adding 1 to each bin
   
 def likelihood_neg(self,index,feature):
  self.no=0
  feature_no=0
  for i in range(0,self.train.shape[0]):
   if(self.train[i,9]==-1):
    self.no=self.no+1
    if(self.train[i,index]==feature):
      feature_no=feature_no+1
  return ((feature_no+1)/(self.no+9))
  
 def posteriorProb(self):
  true_yes=0
  true_no=0
  
  for i in range(0,self.test.shape[0]):
  
   cond_likelihood_neg=1
   cond_likelihood_pos=1
   for j in range(0,8):
    cond_likelihood_pos=cond_likelihood_pos*self.likelihood_pos(j,self.test[i,j])
    cond_likelihood_neg=cond_likelihood_neg*self.likelihood_neg(j,self.test[i,j])  
   post_prob_pos=cond_likelihood_pos*self.yes/self.total
   post_prob_neg=cond_likelihood_neg*self.yes/self.total  
   if(post_prob_pos>=post_prob_neg):
    if(self.test[i,9]==1):
     true_yes=true_yes+1  
   else:
    if(self.test[i,9]==(-1)):
     true_no=true_no+1
  return ((true_yes+true_no)/self.test.shape[0])

  

class LogRegression:
 def __init__(self,file1,frac):
  self.dataset=np.genfromtxt(file1,delimiter=",")
  self.dataset=sc.delete(self.dataset,0,1)
  for i in range(self.dataset.shape[0]): #removig the data entries with null value
   if (np.isnan(self.dataset[i,5])):
    np.delete(self.dataset,i,0)
  np.random.shuffle(self.dataset) # randomize dataset for better accuracy
  for x in range(self.dataset.shape[0]):
   if(self.dataset[x,9]==2):
    self.dataset[x,9]=1
   elif(self.dataset[x,9]==4):
    self.dataset[x,9]=0
  frac=int((frac*self.dataset.shape[0]*2)/(100*3)) # taking only fraction part of 2/3 training set for training
  self.dataset, self.test = self.dataset[:frac,:], self.dataset[467:,:] # Partitioning the dataset into training and test data for cross validation
  self.w=np.zeros((self.dataset.shape[1],1))
  self.classes=self.dataset[:,9]
  self.classes=self.classes.reshape(self.classes.shape[0],1)
 
  
 def preprocess(self): # replcing missing values with mean values for missing data handling
  m=0
  count=0
  for i in range(self.dataset.shape[0]):
   if (not np.isnan(self.dataset[i,5])):
    m=m+self.dataset[i,5]
   count=count+1 
  m=m/count
  m=int(m)
 
  for i in range(self.dataset.shape[0]):
   if (np.isnan(self.dataset[i,5])):
    self.dataset[i,5]=m
   
 def probability(self):
  a=self.dataset.dot(self.w)
  return 1/(1+np.exp(-a))  
  
 def costFun(self):
  p = self.probability()
  loglikelihood = self.classes*np.log(p) + (1-self.classes)*np.log(1-p)
  return -1*loglikelihood.sum()
 
 def gradientAscent(self):
  error=self.classes-self.probability()
  mult=error*self.dataset
  return mult.sum(axis=0).reshape(self.w.shape)
  
 def gradient(self):
  prev_cost=self.costFun()	
  self.tolerance=1e-4
  max_iterations=1000
  iteration=0
  difference=self.tolerance+1
  while ((difference > self.tolerance) and (iteration < max_iterations)):
   self.w=self.w+(1e-4)*self.gradientAscent() # learning rate = 1e-4
   cost=self.costFun()
   difference=np.abs(prev_cost-cost)
   iteration=iteration+1
     
   
 def predict(self):
  label=self.test.dot(self.w)
  good=0
  for i in range(0,label.shape[0]):
   if ((label[i])>0.5):
    label[i]=1
   else:
    label[i]=0  
   if(label[i]==self.test[i,9]):
    good=good+1 
  return good/label.shape[0] 
  
 
if __name__=="__main__":
 x = [1,2,3,12.5,62.5,100] #Percentage of dataset for training the classifier
 accuracy_logi=[]
 accuracy_bayes=[]
 indi_acc_logi=[]
 indi_acc_bayes=[]
 l=len(x)
 for i in range(0,l):
  for j in range(0,5):
   model=LogRegression("breast-cancer-wisconsin.csv",x[i])
   model2=bayesClassify("breast-cancer-wisconsin.csv",x[i])
   model.preprocess()
   model.gradient()
   indi_acc_logi=model.predict()
   indi_acc_bayes=model2.posteriorProb()
  acc_logi=np.mean(indi_acc_logi) # taking average of accuracy of 5 randomized training dataset for given fraction
  acc_bayes=np.mean(indi_acc_bayes)
  accuracy_logi.append(acc_logi)
  accuracy_bayes.append(acc_bayes)
  print 'For Logistic Regression accuracy : %s for %s percentage fraction'%(acc_logi,x[i]) 
  print 'For Bayes classifier accuracy : %s for %s percentage fraction'%(acc_bayes,x[i])
  
 #plotting the graph
 p1, =plt.plot(x, accuracy_logi,label="label 1")
 p2, =plt.plot(x,accuracy_bayes,label="label 2")
 plt.xlabel("Percentage of Considred Training samples")
 plt.ylabel("Accuracy")
 l1 = plt.legend([p1], ["Logistic Regression Classifier"], loc=3)
 l2=plt.legend([p2], ["Bayes Classifier"], loc=4)
 plt.gca().add_artist(l1)
 plt.gca().add_artist(l2)
 plt.show()

