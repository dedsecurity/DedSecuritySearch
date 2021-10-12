#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertModel, BertTokenizer
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import string
import torch
import random

with open('./links.json') as content:
  datali = json.load(content)

tags = []
inputs = []
responses={}
for intent in datali['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])

data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})

data
data = data.sample(frac=1)

data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
data

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
print(input_shape)

vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
output_length = le.classes_.shape[0]
print("output length: ",output_length)

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model = Model(i,x)

model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

train = model.fit(x_train,y_train,epochs=300)

model1 = BertModel.from_pretrained('bert-base-uncased')
tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')

while True:

  import random

  print("\033[32mDed Security Search \033[m")
  texts_p = []
  prediction_input = input('\033[36mSearch: \033[m')

  tokens = tokenizer1.tokenize(prediction_input)
  tokens = ['[CLS]'] + tokens + ['[SEP]']
  tokens = tokens + ['[PAD]'] + ['[PAD]']
  attention_mask = [1 if i!= '[PAD]' else 0 for i in tokens]
  token_ids = tokenizer1.convert_tokens_to_ids(tokens)
  token_ids = torch.tensor(token_ids).unsqueeze(0)
  attention_mask = torch.tensor(attention_mask).unsqueeze(0)
  hidden_rep, cls_head = model1(token_ids, attention_mask = attention_mask)

  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  texts_p.append(prediction_input)
  
  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = pad_sequences([prediction_input],input_shape)
  
  output = model.predict(prediction_input)
  output = output.argmax()
  
  response_tag = le.inverse_transform([output])[0]
  print(random.choice(responses[response_tag]))