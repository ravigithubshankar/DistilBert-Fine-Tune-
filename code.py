import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#install following dependices

import torch
import tensorflow as tf
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
from torch.utils.data import DataLoader,Dataset
import tensorflow as tf
import pandas
from transformers import BertTokenizer, TFBertForSequenceClassification


_____________________________________________________________________________________________________________________________________________________________________________________________________________

#import train.csv files

train_path=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_path=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print(train_path.shape,"\n")
print(test_path.shape,"\n")

#view the target label
#fill the missing values with 0.

labels=train_path["target"]
train_path.fillna("0",inplace=True)
test_path.fillna("0",inplace=True)

#model importation
#pretrained of model of both bert-base-uncased,bert-base-cased
#following  pretrained models from tensorflow and pytorch 

model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
model1=TFBertForSequenceClassification.from_pretrained("bert-base-cased")

token=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model=DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=2)


#We must convert our "Natural Language" to token IDs to train our model. This is done by a Tokenizer, 
#which tokenizes the inputs (including converting the tokens to their corresponding IDs in the pre-trained vocabulary)

train_tokens = tokenizer.batch_encode_plus(
    train_path['text'].tolist(), 
    #max_length=128,  
    padding=True,  
    truncation=True,  
    #return_attention_mask=True,  
    #return_tensors='tf'  
)

test_tokens = tokenizer.batch_encode_plus(
    test_path['text'].tolist(), 
    #max_length=128,  
    padding=True,  
    truncation=True,  
    #return_attention_mask=True,  
    #return_tensors='tf'  
)
#you can change max_length and attention_mask according to your choices

"""
train_tokens1=tokenizer.batch_encode_plus(
train_path["text"].tolist(),
max_length=128,
padding=True,
truncation=True,
return_attention_mask=True,
return_tensors="tf")

test_tokens1=tokenizer.batch_encode_plus(
test_path["text"].tolist(),
max_length=128,
padding=True,
truncation=True,
return_attention_mask=True,
return_tensors="tf")
"""
#above u can see max_length and attention_mask
#train dataset converting into tensor slices
#test dataset converting into test slices

train_dataset1 = tf.data.Dataset.from_tensor_slices((
    dict(train_tokens),
    labels
))
test_dataset1 = tf.data.Dataset.from_tensor_slices((
    dict(test_tokens),
))

train_dataset2=tf.data.Dataset.from_tensor_slices((
dict(train_tokens1),
labels))

test_dataset2=tf.data.Dataset.from_tensor_slices((
dict(test_tokens1)))


train_input_ids = train_tokens['input_ids']
train_attention_masks = train_tokens['attention_mask']
train_labels = labels.values

test_input_ids = test_tokens['input_ids']
test_attention_masks = test_tokens['attention_mask']

#compile the model


#checking the gpu device and enabling


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

#fine-tuning
for epoch in range(num_epochs):
    total_loss=0
    for batch in loader:
        optimizer.zero_grad()
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["label"].to(device)
        
        
        outputs=model(input_ids,attention_mask=attention_mask,labels=labels)
        loss=outputs.loss
        logits=outputs.logits
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        
    print(f"epoch {epoch+1}/{num_epochs},loss:{total_loss:.4f}")

#with out fine-tuning

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=["accuracy"])

print(model.summary())
batch_size=32
epochs=10

#without fine-tuning train the model with batch size of 32

model.fit(train_dataset2.shuffle(1000).batch(32),epochs=epochs)

#evaluate the results

predictions=model.predict(test_dataset2.batch(32))




