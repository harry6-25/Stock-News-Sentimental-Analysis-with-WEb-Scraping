#!/usr/bin/env python
# coding: utf-8

# ## 1. Install and Import Dependencies 

# In[ ]:


#!pip install transformers bs4 


# In[1]:


from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests


# ## 2. Load Summarization Model

# In[4]:


model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)


# ## 3. Summarize a Single Article

# In[8]:


url = "https://finance.yahoo.com/news/i-will-never-cover-game-stop-stock-ever-again-top-analyst-173202488.html"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p')


# In[25]:


text = [paragraph.text for paragraph in paragraphs]
words = ' '.join(text).split(' ')[:350]
ARTICLE = ' '.join(words)


# In[26]:


input_ids = tokenizer.encode(ARTICLE, return_tensors='pt')
output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
summary = tokenizer.decode(output[0],skip_special_token=True)


# In[27]:


summary


# ## 4. Pipeline for News and Sentiment

# In[28]:


monitored_tickers = ['GME', 'TSLA', 'BTC']


# In[34]:


# Search for Stock News using Google and Yahoo Finance
def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=yahoo+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    href = [link['href']for link in atags]
    return href


# In[37]:


raw_urls = {ticker:search_for_stock_news_urls(ticker) for ticker in monitored_tickers}


# In[38]:


# Strip out Unwanted URL
import re


# In[40]:


exclude_list = ['maps','policies','preferences','accounts','support']


# In[44]:


def strip_unwanted_urls(urls, excluded_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list):
            res = re.findall(r'(https?://\S+)',url)[0].split('&')[0]
            val.append(res)
    return list(set(val)) # Avoid Duplicates


# In[45]:


strip_unwanted_urls(raw_urls['GME'],exclude_list)


# In[52]:


cleaned_urls = {ticker:strip_unwanted_url(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}


# In[56]:


# Search and Scrape Cleaned URLs
def scrape_and_process(urls):
    ARTICLES = []
    for url in urls:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES


# In[61]:


articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}


# In[ ]:


# Summarize all Articles


# In[73]:


def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0],skip_special_token=True)
        summaries.append(summary)
    return summaries


# In[76]:


summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}


# In[75]:


summaries['GME']


# ## 5. Sentiment Analysis

# In[78]:


from transformers import pipeline
sentiment = pipeline('sentiment-analysis')


# In[79]:


scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers
    }


# In[80]:


scores


# In[81]:


print(summaries['BTC'][0], scores['BTC'][0]['label'], scores['BTC'][0]['score'])


# ## 6. Export Results to CSV

# In[85]:


def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ]
            output.append(output_this)
    return output


# In[86]:


final_output = create_output_array(summaries, scores, cleaned_urls)


# In[87]:


final_output[0]


# In[88]:


final_output.insert(0,['Ticker','Summary','Label','Confidence','URL'])


# In[90]:


final_output


# In[92]:


import csv
with open('stocksummaries.csv', mode='w', newline='')as f:
    csv_writer = csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)

