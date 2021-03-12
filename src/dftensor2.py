# script dftensor2.py
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

def normalize_df(input_df, mean, std, cols_to_normalize):
    n_df = input_df.copy()
    for item in cols_to_normalize:
        n_df[item] = (n_df[item] - mean[item])/std[item]    
    return n_df

#TRAIN_DATA_URL = "https://storage.googleapis.com/applied-dl/heart.csv"
#TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

#csv_file = tf.keras.utils.get_file("heart.csv",TRAIN_DATA_URL)
#df = pd.read_csv(csv_file)
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
train_df, test_df = train_test_split(normal_df_dummy, test_size = 0.10)
train_target = train_df.pop('target')
test_target = test_df.pop('target')
actual = test_target.to_numpy()

#sys.exit(0)

train_ds = tf.data.Dataset.from_tensor_slices((train_df.values, train_target.values))
#for feat, targ in train_ds.take(6):
#    print('Features {}, Target {}'.format(feat, targ))

batched_train_ds = train_ds.shuffle(len(df)).batch(5)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation = 'relu'),
    tf.keras.layers.Dense(20, activation = 'relu'),
    tf.keras.layers.Dense(1)
    ])
model.compile(optimizer = 'adam',
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
        metrics = ['accuracy']
        )
model.fit(batched_train_ds,epochs = 10)

model.summary()

test_ds = tf.data.Dataset.from_tensor_slices((test_df.values, test_target.values))
batched_test_ds = test_ds.batch(5)
model.evaluate(batched_test_ds, verbose=2)
sys.exit(0)

predictions = model.predict(batched_test_ds)
print(np.shape(predictions))
print(predictions[:5])
predictions_prob = tf.sigmoid(predictions).numpy()
for i in range(10):
    print('predict:{} actual:{}'.format(predictions_prob[i], actual[i]))



