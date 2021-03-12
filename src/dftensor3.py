# script dftensor3.py
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import json


df = pd.read_csv('/home/debdulalm2016/testdata/htrain.csv')

np.set_printoptions(precision=3, suppress=True)
df = df.drop(columns= ['Unnamed: 0'])
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
test_df = df[10:20]

target = df.pop('target')

with open('./models/dftensor1_mod.json',"r") as fp:
    json_string = json.load(fp)
print(type(json_string))
sys.exit(0)

model = tf.keras.models.model_from_json(json_string)
model.load_weights('./weights/dftensor1_wt')

actual = test_df.pop('target')
test_ds = tf.data.Dataset.from_tensor_slices((test_df.values))
batched_test_ds = test_ds.batch(5)

predictions = model.predict(batched_test_ds)
print(np.shape(predictions))
print(predictions)
predictions_prob = tf.sigmoid(predictions).numpy()
print('predicted probability:{}'.format(predictions_prob))
print('actual:{}'.format(actual.to_numpy()))



