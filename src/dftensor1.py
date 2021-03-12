# script dftensor1.py
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import json


#TRAIN_DATA_URL = "https://storage.googleapis.com/applied-dl/heart.csv"
#TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

#csv_file = tf.keras.utils.get_file("heart.csv",TRAIN_DATA_URL)
#df = pd.read_csv(csv_file)
df = pd.read_csv('/home/debdulalm2016/testdata/htrain.csv')

np.set_printoptions(precision=3, suppress=True)
df = df.drop(columns= ['Unnamed: 0'])
df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes
test_df = df[10:20]

target = df.pop('target')
train_ds = tf.data.Dataset.from_tensor_slices((df.values, target.values))
#for feat, targ in train_ds.take(6):
#    print('Features {}, Target {}'.format(feat, targ))
#print(tf.constant(df['thal']))

#sys.exit(0)

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
actual = test_df.pop('target')
test_ds = tf.data.Dataset.from_tensor_slices((test_df.values))
batched_test_ds = test_ds.batch(5)

predictions = model.predict(batched_test_ds)
print(np.shape(predictions))
print(predictions)
predictions_prob = tf.sigmoid(predictions).numpy()
print('predicted probability:{}'.format(predictions_prob))
print('actual:{}'.format(actual.to_numpy()))

#model.save_weights('./weights/dftensor1_wt')
#print('model weights saved in file weights/dftensor1_wt')
#json_string = model.to_json()
#with open('./models/dftensor1_mod.json',"w") as f:
#    json.dump(json_string,f)
#print('model saved in models/dftensor1_mod.json')
