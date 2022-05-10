#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required librairies

import pandas as pd 
import numpy as np

#importing svm from scikit learn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

#import matplotlib library to plot the charts
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#this is the library for statistic data visualization
import seaborn

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("C:\\Users\\asims\\Documents\\Machine Learning and Deep Learning\\creditcard.csv")


# In[3]:


dFrame = pd.DataFrame(data) #Dataframe to PandaDataFrame
dFrame.describe()


# In[4]:


dFrame_fraud = dFrame[dFrame['Class'] == 1] #recovering fraud data
plt.figure(figsize=(15,10)) #assigning figuresize
plt.scatter(dFrame_fraud['Time'], dFrame_fraud['Amount']) #showing fraud amount with respective to it's time
plt.title('Scratter plot for amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0,180000])
plt.ylim([0,2300])
plt.show()


# In[6]:


biggerFraud = dFrame_fraud[dFrame_fraud['Amount'] > 1000].shape[0] #lets do the recovery of frauds more than 1000
print('There are only '+ str(biggerFraud) + ' frauds in total that were bigger than 1000 among ' + str(dFrame_fraud.shape[0]) + ' frauds')


# In[9]:


numOfFrauds = len(data[data.Class == 1])
numOfNoFrauds= len(data[data.Class == 0])
print('There are ' + str(numOfFrauds) + ' frauds in the original dataset, even though there are ' + str(numOfNoFrauds) +' frauds.')
print("\nNow the Accuracy of classifier: "+ str((284315-492)/284315))


# In[10]:


dFrame_corr = dFrame.corr() #Correlation coefficients calculation in pairs with the default method Pearson,Stnd Correlation Coefficient


# In[14]:


plt.figure(figsize=(15,10)) #setting the figure size
seaborn.heatmap(dFrame_corr, cmap="YlGnBu") #heatmap correlation display
seaborn.set(font_scale=2,style='white')
plt.title('Heatmap Correlation Display')
plt.show()


# In[15]:


rank = dFrame_corr['Class'] #Retrieving correlation coefficients as w.r.t feature class
dFrame_rank = pd.DataFrame(rank) 
dFrame_rank = np.abs(dFrame_rank).sort_values(by='Class',ascending=False) #ranking absolute values of the coefficients in DESC 
dFrame_rank.dropna(inplace=True) #removing the missing data but not number


# In[16]:


#we have to divide data in two groups- train dataset & test dataset

#Now build train dataset 
dFrame_train_all = dFrame[0:150000] # start to separate original dataset into frauds & non frauds
dFrameTrainds_1 = dFrame_train_all[dFrame_train_all['Class'] == 1] 
dFrameTrainds_0 = dFrame_train_all[dFrame_train_all['Class'] == 0]
print('In this dataset, we have ' + str(len(dFrameTrainds_1)) +" frauds so we need to take a similar number of non-fraud")

dFrame_sample=dFrameTrainds_0.sample(300)
dFrame_train = dFrameTrainds_1.append(dFrame_sample) #collecting frauds along with no frauds. 
dFrame_train = dFrame_train.sample(frac=1) #Mixing the dataset


# In[19]:


XTrainD = dFrame_train.drop(['Time', 'Class'],axis=1) # We drop the features Time
YTrainD = dFrame_train['Class'] #class as label
XTrainD = np.asarray(XTrainD)
YTrainD = np.asarray(YTrainD)


# In[20]:


#To check if whether the model learn correctly with respect to dataset
TestAll_dFrame = dFrame[150000:]
TestAll_X = TestAll_dFrame.drop(['Time', 'Class'],axis=1)
TestAll_Y = TestAll_dFrame['Class']
TestAll_X = np.asarray(TestAll_X)
TestAll_Y = np.asarray(TestAll_Y)


# In[21]:


XTrainD_rank = dFrame_train[dFrame_rank.index[1:11]] # 1 to 11 takes only 10 features
XTrainD_rank = np.asarray(XTrainD_rank)


# In[22]:


#To check if whether the model learn correctly with respect to dataset 
TestAll_X_rank = TestAll_dFrame[dFrame_rank.index[1:11]]
TestAll_X_rank = np.asarray(TestAll_X_rank)
TestAll_Y = np.asarray(TestAll_Y)


# In[23]:


class_names=np.array(['0','1']) #FYI Class = 1 is fraud and Class = 0 is no fraud, using binary values


# In[25]:


# Create Plot Confusion Matirx Method
def plot_Cmrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True-Label')
    plt.xlabel('Predicted-Label')


# In[26]:


classifier_SVM = svm.SVC(kernel='linear') # We set a SVM classifier, the default SVM Classifier (Kernel = Radial Basis Function)
classifier_SVM.fit(XTrainD, YTrainD) # Then we train our model, with our balanced data train.


# In[27]:


prediction_SVM_all = classifier_SVM.predict(TestAll_X) #And finally, we predict our data test.


# In[28]:


cm = confusion_matrix(TestAll_Y, prediction_SVM_all)
plot_Cmrix(cm,class_names)


# In[29]:


print('Criterion Result that we got is ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[30]:


print('So We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nThe probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("\nAccuracy -> "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[31]:


classifier_SVM.fit(XTrainD_rank, YTrainD) #here we train the model with balanced data train.
prediction_SVM = classifier_SVM.predict(TestAll_X_rank) # Now predict data test.


# In[32]:


cm = confusion_matrix(TestAll_Y, prediction_SVM)
plot_Cmrix(cm,class_names)


# In[33]:


print('Criterion Result that we got is ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[34]:


print('So We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nThe probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("\nAccuracy -> "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[35]:


classifier_SVM_b = svm.SVC(kernel='linear',class_weight={0:0.6, 1:0.4})


# In[36]:


classifier_SVM_b.fit(XTrainD, YTrainD) # Then we train our model, with our balanced data train


# In[37]:


prediction_SVM_b_all = classifier_SVM_b.predict(TestAll_X) #We predict all the data set.


# In[38]:


cm = confusion_matrix(TestAll_Y, prediction_SVM_b_all)
plot_Cmrix(cm,class_names)


# In[39]:


print('Criterion Result that we got is ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[40]:


print('So We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\nThe probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("\nAccuracy -> "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[41]:


classifier_SVM_b.fit(XTrainD_rank, YTrainD) # Now train the model with balanced train data.
prediction_SVM = classifier_SVM_b.predict(TestAll_X_rank) #Now predict data test.


# In[42]:


cm = confusion_matrix(TestAll_Y, prediction_SVM)
plot_Cmrix(cm,class_names)


# In[43]:


print('Criterion Result that we got is ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[44]:


print('So We have detected ' + str(cm[1][1]) + ' frauds / ' + str(cm[1][1]+cm[1][0]) + ' total frauds.')
print('\n The probability to detect a fraud is ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("\nAccuracy -> "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[ ]:




