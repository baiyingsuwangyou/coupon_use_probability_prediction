import pickle
import numpy as np
import pandas as pd

model = open('./model/model.pickle', 'rb')
model = pickle.load(model)
test = pd.read_csv('./preprocess_data/test_f.csv')
# print(test.head())
# test.info()

pred = model.predict(test).tolist()
# print(pred)
# print(len(pred))
# print(set(pred))
test = pd.read_csv('./data/test_offline.csv')
test['predict'] = pred
# print(test.head())
# test.info()
test.to_csv('./predict.csv', index=None)
