# script csvtensor1.py
import numpy as np
import tensorflow as tf
import functools
import sys

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s} : {}".format(key,value.numpy()))
        print(" Labels: {} \n ".format(label.numpy()))

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv",TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv",TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)
LABEL_COLUMN = 'survived'
LABELS = [0,1]

def get_dataset(filepath, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(filepath,batch_size=5,
        label_name=LABEL_COLUMN,
        na_value = "?",
        num_epochs = 1,
        ignore_errors=True,
        **kwargs)
    return dataset

train_ds = get_dataset(train_file_path)
test_ds = get_dataset(test_file_path)
#show_batch(test_ds)

#SELECT_COLUMNS = ['survived', 'age','n_siblings_spouses','parch','fare']
#DEFAULTS = [0,0.0,0.0,0.0,0.0]
#temp_ds = get_dataset(train_file_path,select_columns = SELECT_COLUMNS,column_defaults = DEFAULTS)
#show_batch(temp_ds)
class PackNumericFeatures(object):
    def __init__(self,names):
        self.names = names

    def __call__(self,features,labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat,tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis = -1)
        features['numeric'] = numeric_features
        return features, labels

NUMERIC_FEATURES = ['age','n_siblings_spouses','parch','fare']
packed_train_ds = train_ds.map(PackNumericFeatures(NUMERIC_FEATURES))
packed_test_ds = test_ds.map(PackNumericFeatures(NUMERIC_FEATURES))

#show_batch(packed_train_ds)
#example_batch, example_labels = next(iter(packed_train_ds))

import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
    return (data - mean)/std


normalizer = functools.partial(normalize_numeric_data, mean = MEAN, std = STD)
numeric_column = tf.feature_column.numeric_column(
        'numeric', normalizer_fn = normalizer,
        shape = [len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
#print(example_batch['numeric'])

CATEGORIES = {
        'sex': ['male', 'female'],
        'class': ['First', 'Second', 'Third'],
        'deck': ['A','B','C','D','E','F','G','H','I','J','unknown'],
        'embark_town': ['Cherbourg','Southhampton','Queenstown'],
        'alone': ['y','n']}
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key = feature, vocabulary_list = vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))



preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
#print(preprocessing_layer(example_batch).numpy()[0])

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1)
    ])
model.compile(optimizer = 'adam',
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
        metrics = ['accuracy']
        )
model.fit(packed_train_ds,epochs = 5)

model.summary()

model.evaluate(packed_test_ds)

predictions = model.predict(packed_test_ds)
print(np.shape(predictions))
print(predictions[:5])
predictions_prob = tf.sigmoid(predictions).numpy()
print("predictions probability:")
print(predictions_prob[:5])




