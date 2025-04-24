import os
os.chdir("C:/newpython" )
import pandas as pd
dataframe = pd.read_csv('news.csv')
dataframe.head()
x = dataframe['text']
y = dataframe['label']
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train,y_train)
y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cf)
def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)

import pickle
pickle.dump(classifier,open('model.pkl', 'wb'))
# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))
def fake_news_det1(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    print(prediction)
    fake_news_det1("World's newspapers react to 'Hebdo' attack")