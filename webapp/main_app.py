import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from stqdm import stqdm # https://discuss.streamlit.io/t/stqdm-a-tqdm-like-progress-bar-for-streamlit/10097

import os
import re
from bs4 import BeautifulSoup
from pathlib import Path
import pickle

import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import torch
import transformers
from transformers import pipeline, TFAutoModelForSequenceClassification, AutoTokenizer

##################################################
# Text cleaner
##################################################

def cleaner(text):
	# Remove any HTML tags in-case
	text = BeautifulSoup(text,'lxml').get_text()

	# Type and lower-cased
	text = str(text)
	text = text.lower()

	# Replace "n't" with " not"
	# Doesn’t, Isn’t, Wasn’t, Shouldn’t, Wouldn’t, Couldn’t, Won’t, Can’t, Don’t
	text = re.sub(r"won\'t", "will not", text)
	text = re.sub(r"can\'t", "can not", text)
	text = re.sub(r"n\'t", " not", text)

	# Join negatives "not" or "no" with next word
	text = re.sub("not ", " NOT", text)
	text = re.sub("no ", " NO", text)

	# Remove non-alphabets
	text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)

	# Extra cleaning beyond bs4
	text = re.sub(r"\+", " plus ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ! ", text)
	text = re.sub(r"\?", " ? ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r":", " : ", text)
	text = re.sub(r"\s{2,}", " ", text)

	return text

##################################################
# METHOD 1 - Simple Sentiment Analysis -- Single Review Input
##################################################

@st.cache(allow_output_mutation=True)
def load_tokenizer():
	with open('webapp/cnn_tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	return tokenizer


@st.cache(allow_output_mutation=True)
def load_cnn():
	
	# Download pre-trained neural network if does not exist in folder
	save_dest = Path('webapp/')
	save_dest.mkdir(exist_ok=True)
	model_file = Path("webapp/sentiment_model.hdf5")
	
	if not model_file.exists():
		with st.spinner("Downloading model... this may take awhile! \n Don't close or refresh!"):
			cloud_id = '1Nuc9NEX-CvGMe2w5TG_LrHTcipIWTIGs'
			from gdrive_downloader import download_file_from_google_drive
			download_file_from_google_drive(cloud_id, model_file)
    
	model = load_model(model_file)
	return model

# Predict sentiment (limited to positive/negative)
def predict(user_input):
	cleaned_input = cleaner(user_input)
	padded = pad_sequences(tokenizer.texts_to_sequences([cleaned_input]), 300)
	pred = model.predict(padded)[0][0]
	
	st.write(f'{round(pred*100,2)}% POSITIVE')
	st.write(f'{round((1-pred)*100,2)}% NEGATIVE')
	
	if pred > 0.7:
		st.write('Conclusion: POSITIVE EXPERIENCE')
		st.write("This client has experience an overall positive. Continue to deliver similar products and/or solutions that align with the positives in this review. Try out our in-depth analyzer to further understand your client experiences.")
	elif pred > 0.3:
		st.write('Conclusion: NEUTRAL EXPERIENCE')
		st.write("Although this is not a negative experience, let's discuss about this experience to understand the areas of improvements. Please refer to our consultants at improvements@bestservice.ca. We would like to work with you in developing the best solutions to achieve positive results.")		
	else:
		st.write('Conclusion: NEGATIVE EXPERIENCE')
		st.write("Negative experience are always a great learning opportunity. Let's figure out a solution that can turn this negative into a positive for the future. Please refer to our consultants at improvements@bestservice.ca. We're more than happy to assist you with mitigating negative experiences!")
	
def length(user_input):
	words = user_input.split()
	return len(words)

##################################################
# METHOD 2 - Complex Sentiment Analysis -- Collection of reviews
##################################################

@st.cache(allow_output_mutation=True)
def load_bart():
	# Download pipeline BART model	
	with st.spinner("Preparing analyzer... this may take awhile! \n Don't close or refresh!"):
		model = TFAutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
		tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
		classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
		
	return classifier	
	# PROBLEM, most streamlit apps crash when utilizing torch (pytorch) dependencies
		# Original locally ran model ("facebook/bart-large-mnli") is gigabytes in size
		# https://discuss.streamlit.io/t/getting-error-manager-error-checking-streamlit-healthz-get-http-localhost-8501-healthz/8882/2
		# My delployed app also encounted this issue 
			# "[manager] The service has encountered an error while checking the health of the Streamlit app"
	 	# As per https://discuss.streamlit.io/t/unable-to-deploy-the-app-due-to-the-following-error/6594/10
			# "We currently give 800 MB per app."
		# https://discuss.streamlit.io/t/app-doesnt-deploy-in-streamlit-sharing-possible-issue-with-deploying-pytorch/8007/4
			#  Suggests to use the lightest BART model https://huggingface.co/valhalla/distilbart-mnli-12-3
			# 'valhalla/distilbart-mnli-12-3'
			
def sentiment_pred(text):
	# Possible Sentiment Categories
	# Send the labels and tweets to the classifier pipeline
	candidate_labels = ["positive", "neutral", "negative"]
	result = classifier(text, candidate_labels)
	
	# Extract first label (predicted sentiment)
	# As the classifier results is sorted in descending order (highest one first)
	label = result['labels'][0]
	
	return label

@st.cache(allow_output_mutation=True)
def load_upload():
	df = pd.read_csv(upload)
	with st.spinner("Loaded. Now cleaning..."):
		df.text = df.text.apply(cleaner)
	return df	

########################################################################################################################################################################################################
# WEB-APP
########################################################################################################################################################################################################
st.set_page_config(layout="wide")
page = st.sidebar.selectbox('Select page', ['Please select', 'Simple Analyzer','In-Depth Analyzer'])

if page == 'Please select':
	st.title("REVIEW ANALYSIS CONSULTATION SERVICES")
	st.header("Accelerate your growth.")
	st.markdown("With your client reviews, we can stategically develop innovative solutions to help accelerate positive changes & unlock your vision.")
	
##################################################
# PAGE 1 - SIMPLE, SINGLE REVIEW INPUT
##################################################
# Single review page
elif page == 'Simple Analyzer':
	st.title("REVIEW ANALYSIS CONSULTATION SERVICES")
	
	# Load pre-train tokenizer and neural network model (highest accuracy)
	tokenizer = load_tokenizer()
	model = load_cnn()
	
	st.header('Simple Analyzer')
	st.markdown('Enter a review to analyze!')
	review = st.text_area('15 words minimum')
	
	if not review:
		st.warning('Please enter a review.')
	else:
		if length(review) < 15:
			st.warning('Please enter a minimum 15 word review.')
		else:
			predict(review)

##################################################
# PAGE 2 - COMPLEX, REVIEW COLLECTION INPUT
##################################################
elif page == 'In-Depth Analyzer':
	st.title("REVIEW ANALYSIS CONSULTATION SERVICES")
	
	classifier = load_bart()

	st.header('In-Depth Analyzer')
	st.markdown("Upload your collection of reviews to analyze. _The reviews must be stored into a .csv file under a single column named 'text.'_")
	
	# Sample dataset
	sample_review = pd.read_csv('webapp/movie_review_sample.csv')
	st.download_button("Sample review dataset",
				data=sample_review.to_csv(index=False).encode('utf-8'),
				file_name='movie_review_sample.csv',
				mime='text/csv')
	
	st.markdown("For efficiency purposes, work with 50 reviews at a time.")
	upload = st.file_uploader('Upload a file')
	
	if not upload:
		st.warning('Please upload a .csv file')
	else:
		st.spinner(f'You have selected {upload.name}. Loading file...')
		stqdm.pandas() # initializing progress_apply function
		
		try:
			df = load_upload()
			st.success("Cleaned & stored!")
		except:
			st.write("Your file is not formatted correctly. \n Please ensure that there's a column named 'text' - all lower-cased")
			
		sample_size = st.selectbox('Select sample size', ['Please select', 5, 50, 100, 200, 400, 500, 'ALL'])
		
		################################################## 
		# SENTIMENT DISTRIBUTIONS
		##################################################
		st.header("Sentiment Distribution")
		
		if sample_size == 'Please select':
			st.error('Please select a sample size.')
		elif (sample_size == 'ALL') or (sample_size >= len(df)):
			df2 = df.copy()		
		elif type(sample_size) == int:
			df2 = df.sample(n=sample_size, ignore_index=True)
			
		if len(df2) > 0:
			with st.spinner('Analyzing sentiment...'):
				df2['sentiment'] = df2.text.progress_apply(sentiment_pred) # predict sentiment	
			
			frequency = df2.sentiment.value_counts() # extract sentiment value counts -- which is in descending order
			labels = list(frequency.index) # sentiment labels
			values = list(frequency) # frequencies of each sentiment
		else:
			labels = []
			values = []
			st.error('It seems no analysis was achieved with your uploaded file. Try another file.')
		
		# 2 Columns
		pie, reviews = st.columns(2)
		
		# Column 1 - donut-pie sentiment distirbution
		with pie:	
			# donut pie-chart
			donut = go.Figure(data=[go.Pie(labels=[l.upper() for l in labels], values=values, hole=.3)])
			donut.update_traces(marker = dict(colors=['#b9bfff','#fff9b9','#ffb9bf']))
			st.plotly_chart(donut, use_container_width=True)
				
			if labels[0] == 'positive':
				st.write("The majority of your reviews are __positive__. Let's identify common trends to consitently maintain positivity.")
			elif labels[0] == 'negative':
				st.write("The majority of your reviews are __negative__. Let's identify areas of opportunities for improvement.")
			elif labels[0] == 'neutral':
				st.write("The majority of your reviews are __neutral__. Let's observe the reviews to figure out how we can improve the positives while minimizing the negatives.")
		
		################################################## 
		# LABELED DATAFRAME
		##################################################
		
		# Color mapping created
		color_map = {'negative':'#ffb9e2', 'positive':'#b9ffd6', 'neutral':'#ffd6b9'}
		mapped_colors = list(df2["sentiment"].map(color_map))
		fill_color = [mapped_colors for _ in range(len(df2.columns))]

		# Column 2 - reviews with predicted sentiment
		with reviews:
			table = go.Figure(data=[go.Table(
				header=dict(values=[f"<b>{col.upper()}</b>" for col in list(df2.columns)],
					fill_color='#b9e2ff',
					align='center'),
				cells=dict(values=[df2.text, df2.sentiment],
				       fill_color = fill_color,
				       align='left'))
				])
			table.update_layout(width=1500, height=750)
			st.plotly_chart(table, use_container_width=True)

		
		################################################## 
		# CHARACTER & COUNT DISTRIBUTIONS
		##################################################
		st.header("Character and Word Count Distributions")
		df3 = df.copy()
		char, word = st.columns(2)
		
		with char:	
			# Character count distribution	
			df3['char_count'] = df3.text.str.len()
			fig = px.histogram(df3, x='char_count', color_discrete_sequence=['#ffb9bf'])
			st.plotly_chart(fig, use_container_width=True)
			st.write(f"Most reviews have around a __{df3.char_count.mode()[0]}__ character count, but the overall average is about __{round(df3.char_count.mean())}__ characters per review.")
		
		with word:
			# Word count distribution
			df3['word_count'] = df3.text.str.split().apply(len)
			fig2 = px.histogram(df3, x='word_count', color_discrete_sequence=['#b9bfff'])
			st.plotly_chart(fig2, use_container_width=True)
			st.write(f"Most reviews have around a __{df3.word_count.mode()[0]}__ word count, but the overall average is about of __{round(df3.word_count.mean())}__ words per review.")
		 
		 
			
		
			
		







	   


    

