"""import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Insert the dataset
df_fake = pd.read_csv("C:\Fake news\Fake_news\Fake_news\Fake.csv")
df_true = pd.read_csv("C:\Fake news\Fake_news\Fake_news\True.csv")

# Recognize the real and fake
df_fake["class"] = 0
df_true["class"] = 1

# Remove last 10
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480, 23470, -1):
    df_fake.drop([i], axis=0, inplace=True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416, 21406, -1):
    df_true.drop([i], axis=0, inplace=True)

# Merging fake and real
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("C:\Fake news\Fake_news\Fake_news\manual_testing.csv")

# Merging main true and fake news
df_marge = pd.concat([df_fake, df_true], axis=0)

# Removing date, title, subject
df = df_marge.drop(["title", "subject", "date"], axis=1)
df.isnull().sum()

# Random shuffling
df = df.sample(frac=1)
df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)

# Creating function to convert text into lowercase
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(wordopt)

# Defining dependent and independent variables
x = df["text"]
y = df["class"]

# Splitting dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Convert text into vectors
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic regression
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
LR_score = LR.score(xv_test, y_test)
lr_classification_report = classification_report(y_test, pred_lr)

# Decision tree classification
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT_score = DT.score(xv_test, y_test)
dt_classification_report = classification_report(y_test, pred_dt)

# Gradient boosting classifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
GBC_score = GBC.score(xv_test, y_test)
gbc_classification_report = classification_report(y_test, pred_gbc)

# Random forest classifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
RFC_score = RFC.score(xv_test, y_test)
rfc_classification_report = classification_report(y_test, pred_rfc)

# Model testing with manual entry
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    predictions_message = f"LR Prediction: {output_lable(pred_LR[0])}\n" \
                          f"DT Prediction: {output_lable(pred_DT[0])}\n" \
                          f"GBC Prediction: {output_lable(pred_GBC[0])}\n" \
                          f"RFC Prediction: {output_lable(pred_RFC[0])}"
    messagebox.showinfo("Predictions", predictions_message)

window = tk.Tk()
window.title("Fake News Detection")
window.geometry("400x300")

label = tk.Label(window, text="Enter the news:")
label.pack()

entry = tk.Entry(window)
entry.pack()

def classify_news():
    news = entry.get()
    manual_testing(news)

button = tk.Button(window, text="Classify", command=classify_news)
button.pack()

window.mainloop() 
____________________________________________________________________________________"""
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

vectorization =TfidfVectorizer()
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

def output_label(n):
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
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


   
def process_news():
    news = news_entry.get()
    manual_testing(news)

def browse_file():
    filepath = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("CSV Files", "*.csv"),))
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, filepath)

    if filepath:
        df_fake = pd.read_csv("C:\Fake news\Fake_news\Fake_news\Fake.csv")
        df_true = pd.read_csv("C:\Fake news\Fake_news\Fake_news\True.csv")
        df_fake["class"] = 0
        df_true["class"] = 1
        df_fake_manual_testing = df_fake.tail(10)
        for i in range(23480,23470,-1):
            df_fake.drop([i], axis=0, inplace=True)
        df_true_manual_testing = df_true.tail(10)
        for i in range(21416,21406,-1):
            df_true.drop([i], axis=0, inplace=True)
        df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
        df_marge = pd.concat([df_fake, df_true], axis=0)
        df = df_marge.drop(["title", "subject", "date"], axis=1)
        df["text"] = df["text"].apply(wordopt)
        x = df["text"]
        y = df["class"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        vectorization = TfidfVectorizer()
        xv_train = vectorization.fit_transform(x_train)
        xv_test = vectorization.transform(x_test)
        LR = LogisticRegression()
        LR.fit(xv_train, y_train)
        DT = DecisionTreeClassifier()
        DT.fit(xv_train, y_train)
        GBC = GradientBoostingClassifier(random_state=0)
        GBC.fit(xv_train, y_train)
        RFC = RandomForestClassifier(random_state=0)
        RFC.fit(xv_train, y_train)

        new_def_test = pd.DataFrame({"text": [news]})
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GBC = GBC.predict(new_xv_test)
        pred_RFC = RFC.predict(new_xv_test)

        result = f"LR Prediction: {output_label(pred_LR[0])}\nDT Prediction: {output_label(pred_DT[0])}\nGBC Prediction: {output_label(pred_GBC[0])}\nRFC Prediction: {output_label(pred_RFC[0])}"
        messagebox.showinfo("Prediction Result", result)

root = tk.Tk()
root.title("Fake News Detection")
root.geometry("400x200")

news_label = tk.Label(root, text="Enter the news:")
news_label.pack()

news_entry = tk.Entry(root, width=40)
news_entry.pack()

process_button = tk.Button(root, text="Process", command=process_news)
process_button.pack()

browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack()

file_path_entry = tk.Entry(root, width=40)
file_path_entry.pack()

root.mainloop()
"""___________________________________________________________________________________
from flask import Flask, render_template, request
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

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

def output_label(n):
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
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        news = request.form['news']
        manual_testing_result = manual_testing(news)
        return render_template('index.html', prediction=manual_testing_result)

    return render_template('index.html')

if __name__ == '_main_':
    # Load and preprocess the dataset
    df_fake = pd.read_csv("C:\Fake news\Fake_news\Fake_news\Fake.csv")
    df_true = pd.read_csv("C:\Fake news\Fake_news\Fake_news\True.csv")
    df_fake["class"] = 0
    df_true["class"] = 1
    df_fake_manual_testing = df_fake.tail(10)
    for i in range(23480,23470,-1):
        df_fake.drop([i], axis=0, inplace=True)
    df_true_manual_testing = df_true.tail(10)
    for i in range(21416,21406,-1):
        df_true.drop([i], axis=0, inplace=True)
    df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
    df_marge = pd.concat([df_fake, df_true], axis=0)
    df = df_marge.drop(["title", "subject", "date"], axis=1)
    df["text"] = df["text"].apply(wordopt)
    x = df["text"]
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    GBC = GradientBoostingClassifier(random_state=0)
    GBC.fit(xv_train, y_train)
    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(xv_train, y_train)

    app.run(debug=True)  
    """