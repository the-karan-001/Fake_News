import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
#insert the dataset
df_fake = pd.read_csv("C:\Fake news\Fake_news\Fake_news\Fake.csv")
df_true = pd.read_csv("C:\Fake news\Fake_news\Fake_news\True.csv")
#printing
df_fake.head(5)
df_true.head(5)
#to regonozie the real and fake
df_fake["class"] = 0
df_true["class"] = 1
#to remove last 10
df_fake.shape, df_true.shape
#((23481, 5), (21417, 5))
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)
df_fake.shape, df_true.shape
#((23471, 5), (21407, 5))
#manual testing

#printing the values
df_true_manual_testing.head(10)
df_fake_manual_testing.head(10)
#merging fake and real
df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("C:\Fake news\Fake_news\Fake_news\manual_testing.csv")
#merging main true and fake news
df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)

df_marge.columns
#removing date,title,subject
df = df_marge.drop(["title", "subject","date"], axis = 1)
df.isnull().sum()

#random suffling
df = df.sample(frac = 1)
#printing
df.head()

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)
df.columns
#Index(['text', 'class'], dtype='object')
df.head()
#creating function to convert text into lowercase
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
df["text"] = df["text"].apply(wordopt) 
#defining dependend and independent variables
x = df["text"]
y = df["class"]
#spliting dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#convert text into vectors
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
#logisitic regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train,y_train)
#LogisticRegression()
pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)
#0.9885026737967915 output
#printing
print(classification_report(y_test, pred_lr))
#decision tree classification
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
#decision tree classifier
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
#0.9945632798573975
print(classification_report(y_test, pred_dt))
#gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
#GradientBoostingClassifier(random_state=0)
pred_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)
#printing the values
print(classification_report(y_test, pred_gbc))
#random forest classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
#RandomForestClassifier(random_state=0)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)
#0.9915329768270945
print(classification_report(y_test, pred_rfc))
#Model testing with manual entry
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
   

    prediction = {
        'lr_prediction': output_lable(pred_LR[0]),
        'dt_prediction': output_lable(pred_DT[0]),
        'gbc_prediction': output_lable(pred_GBC[0]),
        'rfc_prediction': output_lable(pred_RFC[0])
    }

    return predictions
   


