#Natural Language Processing
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)#quoting for double quote ignorance

#cleaning the text
import re # to clean the texts
import nltk#nltk is the library which will download stopwords package
nltk.download('stopwords')#stop contains irrelevent words e.g(is,this,the,a,on,of etc)
from nltk.corpus import stopwords#so we remove those stopwords if any in the reviews becoz stopwors will not help to give any hint either the review good or bad
from nltk.stem.porter import PorterStemmer#convert words to root words e.g loved,loving,loves to love
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#a-zA-z is not to remove any letters 
    review=review.lower()# lower the alphsbets in the string
    review=review.split()# conver string of words to list of words
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]#stem only those words which are not stopwords
    review=' '.join(review)#make again from list of words to string of words e.g ['wow','love'] to wow love
    corpus.append(review)

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,1].values
#X=pd.DataFrame(X[:,:],columns=cv.get_feature_names())

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#Fitting Naive Bayes Classifier to the training set

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,Y_train)
#Predicting the test set results

y_pred=classifier.predict(X_test)
#Making the confusion Matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
print(cm)
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
Accuracy=accuracy_score(Y_test,y_pred)*100
print(Accuracy)
Precision=precision_score(Y_test,y_pred)
print(Precision)
Recall=recall_score(Y_test,y_pred)
print(Recall)
F1_Score=f1_score(Y_test,y_pred)
print(F1_Score)
#Accuracy = (TP + TN) / (TP + TN + FP + FN)
#Precision = TP / (TP + FP)
#Recall = TP / (TP + FN)
#F1 Score = 2 * Precision * Recall / (Precision + Recall)