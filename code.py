import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier


#Load the dataset
df = pd.read_csv("dataset.csv",header=None, error_bad_lines=False, encoding= 'unicode_escape')
'''
print(df.head())
print(df.shape)

print(set(df[0]))

'''
from collections import Counter
'''
print(Counter(df[0]))
'''

import re 
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\n", "", string)    
    string = re.sub(r"\r", "", string) 
    string = re.sub(r"[0-9]", "digit", string)
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()
'''
print(df.columns)
'''

#train test split
from sklearn.model_selection import train_test_split
X = []
for i in range(df.shape[0]):
    X.append(clean_str(df.iloc[i][1]))
y = np.array(df[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#feature engineering and model selection
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#pipeline of feature engineering and model
model = Pipeline([('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])

#paramater selection
from sklearn.model_selection import GridSearchCV
parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
               'tfidf__use_idf': (True, False)}


gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X, y)

'''
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)
'''

#preparing the final pipeline using the selected parameters
model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])

#fit model with training data
model.fit(X_train, y_train)
'''
print(model.fit(X_train, y_train))
'''
#evaluation on test data
pred = model.predict(X_test)

'''
print(model.classes_)
'''

from sklearn.metrics import confusion_matrix, accuracy_score

'''
print(confusion_matrix(pred, y_test))
'''
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#print("Accuracy of model is" ,accuracy_score(y_test, pred)*100)

#save the model
from sklearn.externals import joblib
joblib.dump(model, 'model_question_topic.pkl', compress=1)

question = input("Enter the event details")

t=model.predict([question])[0]
print(t)

print("Check output.xls file generated")
#sql

import xlrd
import xlwt
from xlwt import Workbook
loc = ("CCMLEmployeeData.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0,0)

'''
print(sheet.nrows)
print(sheet.ncols)
'''

count=0
j=0
wb1 = Workbook()
sheet1=wb1.add_sheet("Sheet 1", cell_overwrite_ok=True)
l=[]
for i in range(sheet.nrows):
    if(sheet.cell_value(i,2)=='Internships' or sheet.cell_value(i,3)=='Internships'):
        l.append(sheet.cell_value(i,0))
        p=[','.join(l)]
        sheet1.write(1,0,question)
        sheet1.write(1,1,p)
wb1.save('output.xls')

file_errors_location = 'output.xls'
workbook_errors = xlrd.open_workbook(file_errors_location)


