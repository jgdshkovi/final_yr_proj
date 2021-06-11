import requests
#from flask_ngrok import run_with_ngrok
from flask import Flask,render_template,request

import os
import re

import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


import tweepy
from keys import access_token, access_token_secret, ckey, csecret

# assign the values accordingly
consumer_key = ckey
consumer_secret = csecret
access_token = access_token
access_token_secret = access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

##########################################

app = Flask(__name__)
run_with_ngrok(app)

######################################

if torch.cuda.is_available():       
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

def text_preprocessing(text):
	text = re.sub(r'(@.*?)[\s]', ' ', text)
	text = re.sub(r'[0-9]+' , '' ,text)
	text = re.sub(r'\s([@][\w_-]+)', '', text).strip()
	text = re.sub(r'&amp;', '&', text)
	text = re.sub(r'\s+', ' ', text).strip()
	text = text.replace("#" , " ")
	encoded_string = text.encode("ascii", "ignore")
	decode_string = encoded_string.decode()
	return decode_string

def preprocessing_for_bert(data):
	input_ids = []
	attention_masks = []

	# For every sentence...
	for sent in data:
		encoded_sent = tokenizer.encode_plus(
			text=text_preprocessing(sent),  # Preprocess sentence
			add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
			max_length=MAX_LEN,                  # Max length to truncate/pad
			pad_to_max_length=True,         # Pad sentence to max length
			#return_tensors='pt',           # Return PyTorch tensor
			truncation = True,
			return_attention_mask=True      # Return attention mask
			)
		input_ids.append(encoded_sent.get('input_ids'))
		attention_masks.append(encoded_sent.get('attention_mask'))

	# Convert lists to tensors
	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)

	return input_ids, attention_masks

class BertClassifier(nn.Module):
	def __init__(self, freeze_bert=False):
		super(BertClassifier, self).__init__()
		# Specify hidden size of BERT, hidden size of our classifier, and number of labels
		D_in, H, D_out = 768, 50, 2

		# Instantiate BERT model
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		# self.LSTM = nn.LSTM(D_in,D_in,bidirectional=True)
		# self.clf = nn.Linear(D_in*2,2)

		# Instantiate an one-layer feed-forward classifier
		self.classifier = nn.Sequential(
			# nn.LSTM(D_in,D_in)
			nn.Linear(D_in, H),
			nn.ReLU(),
			nn.Linear(H, D_out)
		)

		# Freeze the BERT model
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False
		
	def forward(self, input_ids, attention_mask):
		# Feed input to BERT
		outputs = self.bert(input_ids=input_ids,
							attention_mask=attention_mask)
		# Extract the last hidden state of the token `[CLS]` for classification task
		last_hidden_state_cls = outputs[0][:, 0, :]
		# Feed input to classifier to compute logits
		logits = self.classifier(last_hidden_state_cls)

		return logits

def bert_predict(model, test_dataloader):
	model.eval()
	all_logits = []

	# For each batch in our test set...
	for batch in test_dataloader:
		# Load batch to GPU
		b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

		# Compute logits
		with torch.no_grad():
			logits = model(b_input_ids, b_attn_mask)
		all_logits.append(logits)
	
	# Concatenate logits from each batch
	all_logits = torch.cat(all_logits, dim=0)

	# Apply softmax to calculate probabilities
	probs = F.softmax(all_logits, dim=1).cpu().numpy()

	return probs


# #=======================================================# #

MAX_LEN = 300
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

mdl = BertClassifier()
mdl = torch.load('bert_mdl.pth',map_location=torch.device('cpu'))

# #=======================================================# #

# def tweet_urls(uname):
# 	public_tweets = api.user_timeline(screen_name=uname)
# 	tw_urls = []
# 	txts = []
# 	for tweet in public_tweets:
# 		tw_urls.append('https://twitter.com/{0}/status/{1}'.format(uname,tweet.id))
# 		txts.append(tweet.text)
# 	return tw_urls,txts

def tw_data(uname):
	public_tweets = api.user_timeline(screen_name=uname)
	#DATA = []
	URL = []
	TXT = []
	UN = []
	SCR_NAME = []
	DATE = []
	for tweet in public_tweets[:20]:
		URL.append('https://twitter.com/{0}/status/{1}'.format(uname,tweet.id))
		TXT.append(tweet.text)
		UN.append(tweet.user.name)
		SCR_NAME.append(tweet.user.screen_name)
		DATE.append(tweet.created_at)

	return URL, TXT, UN, SCR_NAME, DATE


@app.route('/', methods=['GET', 'POST'])
def home():
	return render_template('index.html')


@app.route('/features.html')
def features():
	return render_template('features.html')

@app.route('/contact.html')
def contact():
	return render_template('contact.html')

@app.route('/output', methods=['GET', 'POST'])
def output():
	if request.method == 'GET':
		return 'Goto Home & Enter Text There. "GET" not here'
	opt = request.form.get('options')
	if opt=='twt':
		uname = request.form.get('uname')
		thr = request.form.get('thr')

		URL, TXT, UN, SCR_NAME, DATE= tw_data(uname)
		inp, msk = preprocessing_for_bert(TXT)
		tdata = TensorDataset(inp, msk)
		sampler = SequentialSampler(tdata)
		dataloader = DataLoader(tdata, sampler=sampler, batch_size=32)

		probs = bert_predict(mdl, dataloader)

		threshold = int(thr)/100
		preds = np.where(probs[:, 1] > threshold, 1, 0)

		out = []
		for i in range(len(TXT)):
			if preds[i]==1:
				out.append( [TXT[i], SCR_NAME[i], UN[i], URL[i], DATE[i], probs[i][1]] )

		return render_template('twt_out.html',data=out)

	if opt=='tst':
		stxt = request.form.get('stxt')

		inp, msk = preprocessing_for_bert([stxt])
		tdata = TensorDataset(inp, msk)
		sampler = SequentialSampler(tdata)
		dataloader = DataLoader(tdata, sampler=sampler, batch_size=32)

		probs = bert_predict(mdl, dataloader)
		tx_score = probs[0][1]
		out = [stxt]
		out.append(tx_score)
		if tx_score<0.4:
			out.append('Not Toxic')
		elif tx_score<0.75:
			out.append('Mildly Toxic')
		elif tx_score<0.88:
			out.append('Toxic')
		else:
			out.append('Highly Toxic')
		return render_template('tst_out.html', data=[out])


app.run()

