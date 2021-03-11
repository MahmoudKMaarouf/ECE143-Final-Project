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
            


# In[2]:


import pandas as pd

# File paths
file_questions = 'Questions.csv'
file_answers = 'Answers.csv'

# Load dataframes (only loading first 10000 rows for now to reduce processing time)
questions_df = pd.read_csv(file_questions, nrows=10000, encoding = 'iso-8859-1')
answers_df = pd.read_csv(file_answers, nrows=10000, encoding = 'iso-8859-1')


# In[3]:


# Add extra columns to dataframes
# This takes a long time (~10 minutes) to process the entire dataset
# Might be worth exploring the pandas to_pickle() method for saving/loading the dataframes
questions_df['Body_no_tags_no_code']=questions_df['Body'].apply(get_text_no_code)


# In[4]:


questions=questions_df['Body_no_tags_no_code']


# In[5]:


import numpy as np
import seaborn as sns # Import seaborn for visualization
import matplotlib.pyplot as plt
sns.set()


# In[6]:


import nltk # Import natural language toolkit library
nltk.download('stopwords')
import pyLDAvis #Import pyLDAvis for interactive visualization
import pyLDAvis.gensim
import gensim #Import gensim for topic modelling 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy #Import Spacy for lemmatization


# In[7]:


data=list(questions) # Insert the question into list


# In[8]:


# Create bigram and trigram sequences
bigram = gensim.models.Phrases(data, min_count=20, threshold=100) 
trigram = gensim.models.Phrases(bigram[data], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[9]:


import en_core_web_sm  #English pipeline optimized for CPU. Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.
nlp = en_core_web_sm.load()


# In[10]:


# Load tagger
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# get stopwords from nltk fro preprocessing
stop_words = nltk.corpus.stopwords.words('english')

def process_words(texts, stop_words=stop_words, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    """
    Transform the  questions into lowercase, build bigrams-trigrams, and apply lemmatization

    """
    # remove stopwords, short tokens and letter accents 
    texts = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts]
    
    # bi-gram and tri-gram implementation
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []
    
    # implement lemmatization and filter out unwanted part of speech tags
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_tags])
    
    # remove stopwords and short tokens again after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts_out]    
    
    return texts_out


# In[11]:


processed_data = process_words(data) #Apply processing functions to the list


# In[12]:


id2word = corpora.Dictionary(processed_data) # Create dictionary of the words 
print('Vocabulary Size:', len(id2word))


# In[13]:


corpus = [id2word.doc2bow(text) for text in processed_data] #Create Cprpus tuple (BoW format) containing the each word id and their frequency


# In[76]:


#Create dictionary and dataframe of Corpus to remove the high frequncy words 
dict_corpus = {}

for i in range(len(corpus)):
  for idx, freq in corpus[i]:
    if id2word[idx] in dict_corpus:
      dict_corpus[id2word[idx]] += freq
    else:
       dict_corpus[id2word[idx]] = freq
       
dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])


# In[80]:


dict_df


# In[16]:


# Plot histogram of word frequency 
plt.figure(figsize=(8,6))
sns.histplot(dict_df['freq'], bins=100);


# In[73]:


#Top 25 high-frequency words
dict_df2=dict_df.sort_values('freq', ascending=False).head(25) 


# In[74]:


dict_df2


# In[71]:


# Filter out high-frequancy words based on the pre-defined threshold
extension = dict_df[dict_df.freq>1788].index.tolist()


# In[19]:


# List of other non-relevant words identified by inspection that need to be filterd out
unrelevant=['problem', 'good', 'lot', 'people', 'great', 'problem', 'answer', 'question', 'solution', 'wrong', 'prefer', 'mention', 'correctly', 'good', 'easy', 'follow', 'great', 'feel', 'idea', 'recommend', 'support', 'pretty', 'result', 'basic', 'give', 'bad', 'nice', 'try', 'well', 'write', 'look']


# In[20]:


# Add non-relevant words to high-frequncy words
extension.extend(unrelevant)


# In[21]:


# Add high-frequency and nor-relevant words to stop words list
stop_words.extend(extension)
# Rerun the word processing function
processed_data= process_words(data)
# Recreate Dictionary
id2word = corpora.Dictionary(processed_data)
print('New Vocabulary Size:', len(id2word))


# In[22]:


# Filter out words that occur on less than 10 questions, or on more than 60% of the questions.
id2word.filter_extremes(no_below=10, no_above=0.6)
print('Total Vocabulary Size:', len(id2word))


# In[24]:


# Update the corpus 
corpus = [id2word.doc2bow(text) for text in processed_data]


# In[25]:


import os
from gensim.models.wrappers import LdaMallet  # Import Mallet LDA
num_topics=25 # Number of Topics

# Path to Mallet
os.environ['MALLET_HOME'] = 'C:\\new_mallet\\mallet-2.0.8' 
mallet_path = 'C:\\new_mallet\\mallet-2.0.8\\bin\\mallet'


# In[26]:


# Built the topic model using Mallet LDA implementation
ldamallet =gensim.models.wrappers.LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)


# In[27]:


# Print the topics compiled by the model and diplay 5 term and their relative weights
from pprint import pprint
pprint(ldamallet.show_topics(num_words=5,formatted=False))


# In[29]:


# Compute the Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=processed_data, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('Coherence Score: ', coherence_ldamallet)


# In[30]:


# Build another model using multicore LDA implementation and compare the coherence score
from gensim.models import LdaMulticore
ldamulticore = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=id2word,workers=4, eval_every=None, passes=20, batch=True,)


# In[31]:


# Display topics
pprint(ldamulticore.show_topics(num_words=5,formatted=False))


# In[32]:


# Compute Coherence Score for the multicore model
coherence_model_ldamulticore = CoherenceModel(model=ldamulticore, texts=processed_data, dictionary=id2word, coherence='c_v')
coherence_ldamulticore = coherence_model_ldamulticore.get_coherence()
print('Coherence Score: ', coherence_ldamulticore)


# In[39]:


# Build another model using LDA implementation and compare the coherence score with the two previous models
from gensim.models import LdaModel
ldamodel = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, eval_every=None, passes=20,)


# In[40]:


# Display topics
pprint(ldamodel.show_topics(num_words=5,formatted=False))


# In[41]:


# Compute Coherence Score
coherence_model_ldamodel = CoherenceModel(model=ldamodel, texts=processed_data, dictionary=id2word, coherence='c_v')
coherence_ldamodel = coherence_model_ldamodel.get_coherence()
print('Coherence Score: ', coherence_ldamodel)


# In[42]:


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """ function to plot difference between modelsUses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 14))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title)
    plt.colorbar(data)
try:
    get_ipython()
    import plotly.offline as py
except Exception:

    plot_difference = plot_difference_matplotlib
else:
    py.init_notebook_mode()
    plot_difference = plot_difference_plotly


# In[43]:


# Heatmap to compare the correlation between LDA and Multicore LDA
mdiff, annotation = ldamodel.diff(ldamulticore, distance='jaccard', num_words=30)
plot_difference(mdiff, title="LDA vs LDA Multicore Topic difference by Jaccard distance", annotation=annotation)


# In[44]:


# Convert LdaMallet model to a gensim model for Visualization

from gensim.models.ldamodel import LdaModel

def convertldaMalletToldaGen(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha) 
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim


# In[45]:


ldagensim = convertldaMalletToldaGen(ldamallet)


# In[46]:


# Heatmap to compare the correlation between LDA and Multicore LDA
mdiff, annotation = ldagensim.diff(ldamulticore, distance='jaccard', num_words=30)
plot_difference(mdiff, title="LDA Mallet vs LDA Multicore Topic difference by Jaccard distance", annotation=annotation)


# In[47]:


# Get the topic distributions by passing in the corpus to the model
tm_results = ldamallet[corpus]


# In[48]:


# Get the most dominant topic
corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]


# In[49]:


# Get top  significant terms and their probabilities for each topic  
topics = [[(term, round(wt, 3)) for term, wt in ldamallet.show_topic(n, topn=20)] for n in range(0, ldamallet.num_topics)]


# In[52]:


# 5 most probable words for each topic
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], columns = ['Term'+str(i) for i in range(1, 21)], index=['Topic '+str(t) for t in range(1, ldamallet.num_topics+1)]).T
topics_df.head()


# In[53]:


# Display all the terms for each topic
pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, ldamallet.num_topics+1)] )
topics_df


# In[57]:


# Visualize the LDA Mallet terms as wordclouds
from wordcloud import WordCloud # Import wordclouds

# Initiate the wordcloud object
wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [20, 15]

# Create subplots for each topic
for i in range(25):

    wc.generate(text=topics_df["Terms per Topic"][i])
    
    plt.subplot(5, 5, i+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(topics_df.index[i])

plt.show()


# In[55]:


import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(ldagensim, corpus, id2word, sort_topics=False)
pyLDAvis.display(vis_data)


# In[ ]:




