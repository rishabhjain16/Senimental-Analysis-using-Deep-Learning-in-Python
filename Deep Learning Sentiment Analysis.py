# Code copied from : 
# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
# https://www.kaggle.com/ramanchandra/sentiment-analysis-on-imdb-movie-reviews/notebook
# Rishabh Jain


#Importing libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import preprocessing
import scikitplot as skplt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding,LSTM,Dense,Bidirectional

#Load the data
Movie_df=pd.read_csv('IMDB Dataset.csv')
print('Shape of dataset::',Movie_df.shape)
Movie_df.head(10)

#positive and negative reviews
import seaborn as sns
sns.countplot(x='sentiment', data=Movie_df)

#Stats of our data
print("General stats::")
print(Movie_df.info())
print("Summary stats::\n")
print(Movie_df.describe())

#Number of poitive & negative reviews
Movie_df.sentiment.value_counts()

reviews=Movie_df['review']
sentiment=Movie_df['sentiment']

#Summarize no. of classes
print('Classes::\n',np.unique(sentiment))

#Split the data into train & test datasets
train_reviews=reviews[:30000]
train_sentiment=sentiment[:30000]
test_reviews=reviews[30000:]
test_sentiment=sentiment[30000:]
#Shape of train & test dataset
print('Shape of train dataset::',train_reviews.shape,train_sentiment.shape)
print('Shape of test dataset::',test_reviews.shape,test_sentiment.shape)


#Encode our target labels
lb=preprocessing.LabelBinarizer()
#Encode 1 for positive label & 0 for Negative label
train_sentiment=lb.fit_transform(train_sentiment)
test_sentiment=lb.transform(test_sentiment)
#Reshape the array
train_sentiment=train_sentiment.ravel()  
test_sentiment=test_sentiment.ravel()
#Convert categoricals to numeric ones
train_sentiment=train_sentiment.astype('int64')
test_sentiment=test_sentiment.astype('int64')

#Let's explore our data before normalization
train_reviews[0]
test_reviews[30001]

#In above paragraphs, we can observe stopwords,html tags,special charcters & numbers,
# which are not required for sentiment analysis.So we need to remove those by normalizing 
#the review data to reduce dimensionality & noise in the data.
train_sentiment[0:10]
   
test_sentiment[0:10]

#Data Pre-processing

#Let's normalize our data to remove stopwords, html tags and so on.
ps=PorterStemmer()
stopwords=set(stopwords.words('english'))
# Define function for data mining
def normalize_reviews(review):
    #Excluding html tags
    data_tags=re.sub(r'<[^<>]+>'," ",review)
    #Remove special characters/whitespaces
    data_special=re.sub(r'[^a-zA-Z0-9\s]','',data_tags)
    #converting to lower case
    data_lowercase=data_special.lower()
    #tokenize review data
    data_split=data_lowercase.split()
    #Removing stop words
    meaningful_words=[w for w in data_split if not w in stopwords]
    #Appply stemming
    text= ' '.join([ps.stem(word) for word in meaningful_words])
    return text

#Normalize the train & test data
norm_train_reviews=train_reviews.apply(normalize_reviews)
norm_test_reviews=test_reviews.apply(normalize_reviews)

#Let's look at our normalized data
norm_train_reviews[0]
norm_test_reviews[30001]


#Let's create features using bag of words model
cv=CountVectorizer(ngram_range=(1,2))
train_cv=cv.fit_transform(norm_train_reviews)
test_cv =cv.transform(norm_test_reviews)
print('Shape of train_cv::',train_cv.shape)
print('Shape of test_cv::',test_cv.shape)

#Our train & test dataset contains 1929440 attributes each.
#Let's build our traditional ML models
#Random Forest model
#Training the classifier
rfc=RandomForestClassifier(n_estimators=20,random_state=42)
rfc=rfc.fit(train_cv,train_sentiment)
score=rfc.score(train_cv,train_sentiment)
print('Accuracy of trained model is ::',score)

#Making predicitions
rfc_predict=rfc.predict(test_cv)

#How accuate our model is?
cm=confusion_matrix(test_sentiment,rfc_predict)
#plot our confusion matrix
skplt.metrics.plot_confusion_matrix(test_sentiment,rfc_predict,normalize=False,figsize=(12,8))
plt.show()

#print classification report for performance metrics
cr=classification_report(test_sentiment,rfc_predict)
print('Classification report is::\n',cr)

# ROC curve for Random Forest Classifier
fpr_rf,tpr_rf,threshold_rf=roc_curve(test_sentiment,rfc_predict)
#Area under curve (AUC) score, fpr-False Positive rate, tpr-True Positive rate
auc_rf=auc(fpr_rf,tpr_rf)
print('AUC score for Random Forest classifier::',np.round(auc_rf,3))

#Recurrent neural network (RNN) with LSTM (Long Short Term Memory) model
#Train dataset
X_train=train_cv
X_train=[str(x[0]) for x in X_train]
y_train=train_sentiment
# Test dataset
X_test=test_cv
X_test=[str(x[0]) for x in X_test]
y_test=test_sentiment

# Tokenize the train & test dataset
Max_Review_length=500
tokenizer=Tokenizer(num_words=Max_Review_length,lower=False)
tokenizer.fit_on_texts(X_train)
#tokenizig train data
X_train_token=tokenizer.texts_to_sequences(X_train)
#tokenizing test data
X_test_token=tokenizer.texts_to_sequences(X_test)

#Truncate or pad the dataset for a length of 500 words for each review
X_train=pad_sequences(X_train_token,maxlen=Max_Review_length)
X_test=pad_sequences(X_test_token,maxlen=Max_Review_length)

print('Shape of X_train datset after padding:',X_train.shape)
print('Shape of X_test dataset after padding:',X_test.shape)

# Most poplar words found in the dataset
vocabulary_size=5000 
embedding_size=64
model=Sequential()
model.add(Embedding(vocabulary_size,embedding_size,input_length=Max_Review_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))
model.summary()
#Compile our model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#Train our model
batch_size=128
num_epochs=6
X_valid,y_valid=X_train[:batch_size],train_sentiment[:batch_size]
X_train1,y_train1=X_train[batch_size:],train_sentiment[batch_size:]
# Fit the model
history = model.fit(X_train1,y_train1,validation_data=(X_valid,y_valid),validation_split=0.2,
          batch_size=batch_size,epochs=num_epochs, verbose=1,shuffle=True)

# Predictions
y_predict_rnn=model.predict(X_test)
#Changing the shape of y_predict to 1-Dimensional
y_predict_rnn1=y_predict_rnn.ravel()
y_predict_rnn1=(y_predict_rnn1>0.5)
y_predict_rnn1[0:10]

#Accuracy
score = model.evaluate(X_test,y_test)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

#Confusion matrix for RNN with LSTM
cm_rnn=confusion_matrix(y_test,y_predict_rnn1)
#plot our confusion matrix
skplt.metrics.plot_confusion_matrix(y_test,y_predict_rnn1,normalize=False,figsize=(12,8))
plt.show()

#Classification report for performance metrics
cr_rnn=classification_report(y_test,y_predict_rnn1)
print('The Classification report is::\n',cr_rnn)

#ROC curve for RNN with LSTM
fpr_rnn,tpr_rnn,thresold_rnn=roc_curve(y_test,y_predict_rnn)
#AUC score for RNN
auc_rnn=auc(fpr_rnn,tpr_rnn)
print('AUC score for RNN with LSTM ::',np.round(auc_rnn,3))


#PLOT for accuracy and loss
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Receiver Operating Characterstic (ROC) Curve for Model Evaluation
#Now, let's plot the ROC for both Random Forest Classifier & RNN with LSTM
plt.figure(1)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_rnn,tpr_rnn,label='RNN(area={:.3f})'.format(auc_rnn))
plt.plot(fpr_rf,tpr_rf,label='Random Forest (area={:.3f})'.format(auc_rf))
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#Model Evaluation on unseen dataset
Model_evaluation=pd.DataFrame({'Model':['Random Forest Classifier','RNN with LSTM'],
                              'f1_score':[0.81,0.79],
                              'roc_auc_score':[np.round(auc_rf,3),np.round(auc_rnn,3)]})
Model_evaluation

