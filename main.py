import streamlit as st
from config import *
import predictor
import os,wget
from preprocess import clean_title
from transformers import AutoTokenizer
import torch

@st.cache
def load_model():
  if not os.path.exists('headline_model.pt'):  
    site_url = 'https://headline-model.s3.amazonaws.com/headline_model.pt'
    file_name = wget.download(site_url)
  return predictor.PythonPredictor()

for category in categories:
  st.sidebar.write(category)

st.title('Bangla News Headline Category Detector')
st.text('\n')
text = st.text_area('Input Article Title')
clicked = st.button('Detect')
st.text('\n')

if clicked:
  predict_clickbait = load_model()
  text = clean_title(text)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
  input_mask_array = [int(token_id > 0) for token_id in input_ids]                          
  # Convertion to Tensor
  input_ids = torch.unsqueeze(torch.tensor(input_ids),0) 
  input_mask_array = torch.unsqueeze(torch.tensor(input_mask_array),0)
  label_index,probability,probs = predict_clickbait.predict(input_ids,input_mask_array)
  st.subheader(f'Category is : {categories[label_index]}')
  st.text(f'Predicted with {probability*100:.2f}% confidence.')
  st.subheader('Other Predicitons:\n')
  print(probs)
  for i in range(len(categories)):
    if i!=label_index:
      st.text(f'{categories[i]} : {probs[0][i]*100:.2f}%')
