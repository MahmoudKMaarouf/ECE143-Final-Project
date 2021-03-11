#!/usr/bin/env python
# coding: utf-8

# In[1]:


from io import StringIO
from html.parser import HTMLParser     
        
class StackOverflowParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.strict = False # If any invalid html is encountered, parser will make a best guess at its intention
        self.convert_charrefs= True # Hold data section until next tag is encountered
        
        # Field variable to keep track of parsed data with tags removed
        self.text = StringIO()
        self.text_no_code = StringIO()
        
        # Field variables to keep track of and store <code></code> blocks
        self.code_blocks = []
        self.lasttag = None
        
    def handle_starttag(self, tag, attrs):
        '''
        Method inherited from HTMLParser super class that is called whenever the start of a tag is encountered.
        In this parser, it keeps track of the last start tag that was encountered.
        :param tag: Current tag being parsed (ex: p, div, code, etc.)
        :type tag: str
        :param attrs: List of (name,value) pairs containing attributes found inside the tag's brackets
        :type attrs: list[str]
        '''
        assert isinstance(tag,str)
        assert isinstance(attrs, list) 
        
        self.lasttag = tag
        
    def handle_data(self, data): 
        '''
        Method inherited from HTMLParser super class that is called whenever data inside of a tag is encountered.
        In this parser, it saves blocks of code to the field variable self.code and records all text with HTML tags removed
        :param data: Current data inside of a tag being parsed
        :type tag: str
        '''
        assert isinstance(data,str)
        
        # If the last tag encountered was a <code> tag, append the contents to the list of code blocks
        if self.lasttag == "code":
            self.lasttag = None
            self.code_blocks.append(data)
        else:  
            self.text_no_code.write(data)
            
        # Record text between tags
        self.text.write(data)
        
    def get_data(self):
        '''
        Returns parsed text without HTML tags 
        :return: Text wihtout tags
        :type return: str
        '''
        return self.text.getvalue()
    
    def get_data_no_code(self):
        '''
        Returns parsed text without HTML tags and with code blocks removed
        :return: Text wihtout tags
        :type return: str
        '''
        return self.text_no_code.getvalue()
    
def strip_tags(html):
    '''
    Takes in a body of text that is formatted in HTML and returns the same text with the HTML tags now removed. 
    This method bundles the process of instantiating a parser, feeding the data, and returning the parsed output.
    :param html: HTML-formatted body of text
    :type html: str
    :return: The input text now without HTML tags
    :type return: str
    '''
    assert isinstance(html,str)
    
    # Feed text into parser and return parsed text without tags
    s = StackOverflowParser()
    s.feed(html)
    return s.get_data()

def get_text_no_code(html):
    '''
    Takes in a body of text that is formatted in HTML and returns the same text with the HTML tags and blocks of code now removed. 
    This method bundles the process of instantiating a parser, feeding the data, and returning the parsed output.
    :param html: HTML-formatted body of text
    :type html: str
    :return: The input text now without HTML tags or code blocks
    :type return: str
    '''
    assert isinstance(html,str)
    
    # Feed text into parser and return parsed text without tags
    s = StackOverflowParser()
    s.feed(html)
    return s.get_data_no_code()

def get_code(html):
    '''
    Takes in a body of text that is formatted in HTML and returns the blocks of code found within the text. 
    This method bundles the process of instantiating a parser, feeding the data, and returning the blocks of code.
    An empty list is returned if no <code> tags are found.
    :param html: HTML-formatted body of text
    :type html: str
    :return: List of blocks of code found within text
    :type return: list[str]
    '''
    assert isinstance(html,str)
    
    s = StackOverflowParser()
    s.feed(html) 
    return [item.replace('\n', ' ') for item in s.code_blocks]
            


# In[ ]:


from kaggle.api.kaggle_api_extended import KaggleApi

# Import the dataset directly from Kaggle 
# Requires a Kaggle account linked to an API key on your device 
api = KaggleApi()
api.authenticate()
api.dataset_download_files('stackoverflow/pythonquestions', path='./', unzip=True)


# In[2]:


import pandas as pd

# File Paths 
file_questions = 'Questions.csv'
file_answers = 'Answers.csv'
file_tags = 'Tags.csv'

dates = ["CreationDate"]

# Load dataframes (only loading first 10000 rows for now to reduce processing time)
questions_df = pd.read_csv(file_questions, encoding = 'iso-8859-1', nrows=10000, parse_dates=dates)
answers_df = pd.read_csv(file_answers, encoding = 'iso-8859-1', nrows=10000, parse_dates=dates)
tags_df = pd.read_csv(file_tags, encoding = 'iso-8859-1', nrows=10000)


# In[3]:


# Add extra columns to dataframes
# This takes a long time (~10 minutes) to process the entire dataset
# Might be worth exploring the pandas to_pickle() method for saving/loading the dataframes

questions_df = questions_df.fillna('')
#questions_df['Body_no_tags']=questions_df['Body'].apply(strip_tags)
questions_df['Body_no_tags_no_code']=questions_df['Body'].apply(get_text_no_code)
#questions_df['Body_code']=questions_df['Body'].apply(get_code)

#answers_df['Body_no_tags']=answers_df['Body'].apply(strip_tags)
answers_df['Body_no_tags_no_code']=answers_df['Body'].apply(get_text_no_code)
#answers_df['Body_code']=answers_df['Body'].apply(get_code)


# In[10]:


# Creates one database with every question linked to every answer
# For questions with no answer the values are NaN
suffixes = ['.q', '.a']
excess_columns = ['OwnerUserId.q', 'Id.a', 'OwnerUserId.a','ParentId', 'Body.q','CreationDate.a', 'Body.a']
QA_df = questions_df.merge(answers_df, how='left', left_on='Id', right_on='ParentId', suffixes=suffixes).drop(excess_columns , axis=1)

# Merges the questions/answers with all the associated tags 
QAT_df = QA_df.merge(tags_df, how='left', left_on='Id.q', right_on='Id', suffixes=['', '.t']).drop('Id', axis=1)
QAT_df = QAT_df.fillna('')

# Groups all the questions together with a list of all the answers scores and body
# Includes the tags as well 
# Uses a set- might be an issue for pairing the answer scores to the answers
QAT_list = QAT_df.groupby('Id.q').agg(Id=('Id.q', 'max'),
                                      CreationDate=('CreationDate.q', 'max'),
                                      Q_Score=('Score.q', 'mean'),
                                      Title=('Title', 'max'),
                                      Q_Body=('Body_no_tags_no_code.q', 'max'),
                                      A_Score=('Score.a', lambda x: set(x)),
                                      A_Body=('Body_no_tags_no_code.a', lambda x: set(x)),
                                      Tags=('Tag', lambda x: set(x)))
QAT_list.A_Body = QAT_list.A_Body.apply(lambda s: ' '.join(s)) # Turns the set of Answers into a str


# In[11]:


# Filtering out certain questions
QAT_list.iloc[0]['A_Body']


# In[ ]:


# Creates one database with every question linked to every answer
# Questions with no answers are dropped
QA_df = questions_df.merge(answers_df, how='right', left_on='Id', right_on='ParentId', suffixes=suffixes).drop(excess_columns , axis=1)
QAT_df = QA_df.merge(tags_df, how='left', left_on='Id.q', right_on='Id', suffixes=['', '.t']).drop('Id', axis=1)


# In[ ]:


# Row 11 in Questions.csv is a good example with a few code blocks
body = questions_df['Body'][11]
body_no_tags = questions_df['Body_no_tags'][11]
body_code = questions_df['Body_code'][11]
body_no_tags_no_code = questions_df['Body_no_tags_no_code'][11]


# In[ ]:


print(body)


# In[ ]:


print(questions_df)


# In[ ]:


print(body_no_tags_no_code)


# In[ ]:


print(body_code)

