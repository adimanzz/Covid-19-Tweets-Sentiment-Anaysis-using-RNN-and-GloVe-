# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:11:39 2020

@author: aditya
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer


class CleanText():
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.custom = self.stop_words + list(punctuation)
        self.wordnet_lemmatizer = WordNetLemmatizer()
        
    def clean(self, text):
        self.text = text
        self.text = self.text.lower()
        tokens = nltk.tokenize.word_tokenize(self.text)
        tokens = [t for t in tokens if len(t) > 2] #Remove single characters
        tokens = [self.wordnet_lemmatizer.lemmatize(t) for t in tokens] #Lemmatize words
        tokens = [t for t in tokens if t not in self.custom] #Remove Stopwords and Punctuation
        tokens = [t for t in tokens if not any(c.isdigit() for c in t)] #Remove digits
        return tokens


















