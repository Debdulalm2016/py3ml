# script knnclass1.py
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def normalize_df(input_df, mean, std, cols_to_normalize):
    n_df = input_df.copy()
    for item in cols_to_normalize:
        n_df[item] = (n_df[item] - mean[item])/std[item]    
    return n_df

df = pd.read_csv('/home/debdulalm2016/testdata/htrain.csv')
np.set_printoptions(precision=3, suppress=True)
print('Number of rows: {}'.format(len(df)))

df = df.drop(columns= ['Unnamed: 0'])

#target = df.pop('target')
desc = df.describe()
#print(desc)
MEAN = desc.T['mean']
STD = desc.T['std']
#print(MEAN)
#print(STD)
#print("mean and std of age")
#print(MEAN['age'], STD['age'])
age_labels = ['0-40','40-60','above 60']
age_bins = [0,40,60,100]
df['age'] = pd.cut(df['age'],bins=age_bins, labels=age_labels)
print(df.head())
COLS_TO_NORMALIZE = ['cp','trestbps','chol','restecg','thalach','oldpeak','slope','ca']
normal_df = normalize_df(df, MEAN, STD, COLS_TO_NORMALIZE)
print("normalized training data")
print(normal_df.head())
normal_df_dummy = pd.get_dummies(normal_df,columns=['age','thal'])
print("after category conversion")
print(normal_df_dummy.head())
#test_df = normal_df_dummy[10:20]
train_df, test_df = train_test_split(normal_df_dummy, test_size = 0.10,random_state=0)
train_target = train_df.pop('target')
test_target = test_df.pop('target')
actual = test_target.to_numpy()

knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
knn.fit(train_df,train_target)
predictions = knn.predict(test_df)
print('Confusion Matrix:')
print(confusion_matrix(predictions, actual))



