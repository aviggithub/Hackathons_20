# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:10:29 2019

@author: avi
"""

import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
plt.style.use('ggplot')
%matplotlib inline
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

file_Read_train='D:\\Python project\\Tag recommendation\\new_dataset\\train.csv'
file_Read_test='D:\\Python project\\Tag recommendation\\new_dataset\\test.csv'

df_train = pd.read_csv(file_Read_train)
df_test = pd.read_csv(file_Read_test)

df_train.head(5)

data=df_train.copy()

tagc=data['tags'].value_counts()

tagc.dtype

data['tags'].shape


c = data['tags'].astype('category')


data['target'] = c.cat.codes

d = dict(enumerate(c.cat.categories))
#print (d)
data['level_back'] = data['target'].map(d)

#data=data.drop(['target1','targetX','level_back'],axis=1)

data.head(2)


#codes, cats = pd.factorize(data['tags'])

#c = data.tags.astype('category')
#codes = c.cat.codes
#cats = c.cat.categories

#plot bar plot using value count


#data['target2']= [yxx[item] for item in data['Complaint-Status']]
 
data['tiltle_hed']=data['title']+data['article']

def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

stopwords=get_stop_words("D:\\Python project\\stopwords_en.txt")

stop = stopwords.words('english')

data['tiltle_hed']=data['tiltle_hed'].str.replace("  "," ")
data['tiltle_hed']=data['tiltle_hed'].str.replace("  "," ")
data['tiltle_hed'] = data['tiltle_hed'].str.replace('&lt;', '') 
data['tiltle_hed'] = data['tiltle_hed'].str.replace('&gt;', '') 
data['tiltle_hed'] = data['tiltle_hed'].str.replace('<[^<]+?>', '') 
data['tiltle_hed'] = data['tiltle_hed'].str.replace(r'\r\n', '')
data['tiltle_hed'] = data['tiltle_hed'].str.replace(r'\r', '')
data['tiltle_hed'] = data['tiltle_hed'].str.replace(r"[^a-zA-Z]+", ' ')
data['tiltle_hed']= data['tiltle_hed'].str.lower()
data['tiltle_hed']= data['tiltle_hed'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['tiltle_hed'] = data['tiltle_hed'].apply(lambda x: " ".join([x for x in x.split() if len(x)>2]))



df_test['tiltle_hed']=df_test['title']+df_test['article']
data['num_words'] = data['tiltle_hed'].apply(lambda x : len(x.split()))

bins=[0,50,75, np.inf]
data['bins']=pd.cut(data.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])

word_distribution = data.groupby('bins').size().reset_index().rename(columns={0:'counts'})

word_distribution.head()

sns.barplot(x='bins', y='counts', data=word_distribution).set_title("Word distribution per bin")

data.head()

num_class = len((data['tags'].value_counts()))
#y = data['target'].values
y=data['target'].values
len(y)

MAX_LENGTH = 500
tokenizer = Tokenizer()
tokenizer2 = Tokenizer()

tokenizer.fit_on_texts(data['tiltle_hed'].values)
post_seq = tokenizer.texts_to_sequences(data['tiltle_hed'].values)

df_test['tiltle_hed'].shape

tokenizer2.fit_on_texts(df_test['tiltle_hed'].values)
post_seq_test = tokenizer2.texts_to_sequences(df_test['tiltle_hed'].values)

len(post_seq_test)

post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)

len(post_seq_padded)
post_seq_padded_test = pad_sequences(post_seq_test, maxlen=MAX_LENGTH)

len(post_seq_padded_test)

X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.05)

vocab_size = len(tokenizer.word_index) + 1

inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = Dense(32, activation='relu')(x)

predictions = Dense(num_class, activation='softmax')(x)

model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
 model.add(Dense(1, activation='sigmoid'))
batch_size=8


model.summary()
filepath="weights-simple.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=16, y=to_categorical(y_train), verbose=1, validation_split=0.25, 
          shuffle=True, epochs=1)


df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')


predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
accuracy_score(y_test, predicted)

predicted3 = model.predict(post_seq_padded_test)

accuracy_score(y_test, predicted)


predicted3 = np.argmax(predicted3, axis=1)

len(predicted3)
df_test['Com']=predicted3

df_test['Com'].value_counts()

df_test['tags'] = df_test['Com'].map(d)

data.head(2)

nfiless=data[['id','tiltle_hed','tags']]

#df_test.drop('Complaint-Status', axis=1, inplace=True)

df_test.rename(columns={'Comp' :'tags'}, inplace=True)

df_test_output= df_test[['id','tags']]
#df_test_output.dtypes
df_test_output.to_csv("D:\\Python project\\Tag recommendation\\new_dataset\\output.csv", index=False, header=True)

