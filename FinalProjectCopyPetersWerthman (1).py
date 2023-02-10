#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Data Analysis Using Random Forest, Artificial Neural Network, and Gridsearch #
# 
# ## Jon Werthman and Tom Peters ##
# 

# In[41]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
sns.set(style='white', context='notebook', palette='deep')
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler
import os
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[42]:


columns=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
os.path.join(os.getcwd(), 'winequality-red.csv')
wine = pd.read_csv("winequality-red.csv",sep = ';', usecols=columns)
#prints out the number of wines in each quality category
wine.quality.value_counts()


# ## Correlation Grid of Features ##

# In[43]:


plt.figure(figsize=(10, 10))
sns.heatmap(wine.corr(method='pearson'), annot=True, square=True)
plt.show()

print('Correlation of different features of our dataset with quality:')
for i in wine.columns:
  corr, _ = pearsonr(wine[i], wine['quality'])
  print('%s : %.4f' %(i,corr))


# In[44]:


wine = wine.drop('residual sugar', axis = 1)
wine = wine.drop('fixed acidity', axis = 1)
wine = wine.drop('free sulfur dioxide', axis = 1)


# # Updated Heat Map With Dropped Columns

# In[45]:


sns.heatmap(wine.corr(method='pearson'), annot=True, square=True)
wine.shape


# In[22]:


#new dataset that has dropped the categories least correlated with quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
wine['quality'] = wine['quality'].map({'bad' : 0, 'good' : 1})
wine.head(15)


# In[23]:


X = wine.drop('quality', axis = 1)
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[24]:





import itertools

#confusion matrix function for data visualizations

def plot_confusion_matrix(cm, classes, normalize = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens): # can change color 
    plt.figure(figsize = (5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
             plt.text(j, i, format(cm[i, j], fmt), 
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)


# ## RFC ##

# In[25]:


rfc = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=0,)
rfc.fit(X_train, y_train)
pred_rf = rfc.predict(X_test)
Y_compare_rfc = pd.DataFrame({'Actual' : y_test, 'Predicted' : pred_rf})
print(Y_compare_rfc.head())
print('\nConfussion matrix:')
print(confusion_matrix(y_test, pred_rf))


cm = confusion_matrix(y_test, pred_rf)
plot_confusion_matrix(cm, classes = ['0 - Bad', '1 - Good'],
                      title = 'Confusion Matrix')


# In[38]:


ANN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)
ANN.fit(X_train, y_train)
pred_ANN = ANN.predict(X_test)
Y_compare_ANN = pd.DataFrame({'Actual' : y_test, 'Predicted' : pred_ANN})

print('\nConfussion matrix:')
print(confusion_matrix(y_test, pred_ANN))


cm = confusion_matrix(y_test, pred_ANN)
plot_confusion_matrix(cm, classes = ['0 - Bad', '1 - Good'],
                      title = 'Confusion Matrix')


# In[ ]:





# ## SVM ##

# In[39]:


svm = SVC(kernel = 'rbf', degree = 10, random_state = 1)
svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

Y_compare_svm = pd.DataFrame({'Actual': y_test, 'Predicted': pred_svm})


print('\nConfusion Matrix: ')
print(confusion_matrix(y_test, pred_svm))
#print(classification_report(y_test, pred_svc))

cm = confusion_matrix(y_test, pred_svm)


plot_confusion_matrix(cm, classes = ['0 - Bad', '1 - Good'],
                     title = 'Confusion Matrix')


# In[28]:


#calculating the best parameters to start with that give the best accuracy
from sklearn.model_selection import GridSearchCV
c = [.05,.1,1,2,5,10,50,100]

parameters = {
    'C': [.05,.1,1,2,5,10,50,100],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
update = GridSearchCV(svm, param_grid=parameters, scoring='accuracy', cv = 5)
update.fit(X_train, y_train)
update.best_params_
update.score(X_test, y_test)

print("The best parameters are %s \nwith a score of %0.5f" % (update.best_params_, update.best_score_))


# In[29]:


#update weights

svmFix = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svmFix.fit(X_train, y_train)
pred_update2 = svmFix.predict(X_test)
cm = confusion_matrix(y_test, pred_update2)
plot_confusion_matrix(cm, classes = ['0 - Bad', '1 - Good'],
                     title = 'Confusion Matrix')


# In[30]:


from sklearn.metrics import mean_squared_error
# error calculations
clf = SVC(kernel='rbf', gamma=0.0017782794100389228,C=10.0)
c = clf.fit(X_train,y_train)

print('Classifier score: ', clf.score(X_test,y_test))
y = c.predict(X_test)
print('Classifier MSE: ',mean_squared_error(y,y_test))


# In[31]:


#calculating accuracy
modelNames = ['Random Forrest', 'Artificial Neural Network', 'Support Vector Model']
modelClassifiers = [rfc, ANN, svm]
models = pd.DataFrame({'modelNames' : modelNames, 'modelClassifiers' : modelClassifiers})
counter=0
score=[]
for i in models['modelClassifiers']:
  accuracy = cross_val_score(i, X_train, y_train, scoring='accuracy', cv=10)
  print('Accuracy of %s Classification model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
  score.append(accuracy.mean())
  counter+=1


# In[ ]:





# In[ ]:




