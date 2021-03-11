#!/usr/bin/env python
# coding: utf-8

# In[ ]:


QAT_list=QAT_list.values.tolist() # Insert QAT_list into list
QAT_list


# In[ ]:


import numpy as np
import seaborn as sns # Import seaborn for visualization
import matplotlib.pyplot as plt
sns.set()


# In[ ]:


import nltk # Import natural language toolkit library
nltk.download('stopwords')
import pyLDAvis #Import pyLDAvis for interactive visualization
import pyLDAvis.gensim
import gensim #Import gensim for topic modelling 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

try:
    import spacy #Import Spacy for lemmatization
    from scipy.sparse.sparsetools.csr import _csr

except:
    from scipy.sparse import sparsetools as _csr


# In[ ]:


# Create bigram and trigram sequences
bigram = gensim.models.Phrases(QAT_list, min_count=20, threshold=100) 
trigram = gensim.models.Phrases(bigram[QAT_list], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[ ]:


import en_core_web_sm  #English pipeline optimized for CPU. Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.
nlp = en_core_web_sm.load()


# In[ ]:


# Load tagger
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# get stopwords from nltk fro preprocessing
stop_words = nltk.corpus.stopwords.words('english')

def process_words(texts, stop_words=stop_words, allowed_tags=['NOUN', 'VERB']):
    
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


# In[ ]:


processed_data = process_words(QAT_list) #Apply processing functions to the list
processed_data


# In[ ]:


id2word = corpora.Dictionary(processed_data) # Create dictionary of the words 
print('Vocabulary Size:', len(id2word))


# In[ ]:


corpus = [id2word.doc2bow(text) for text in processed_data] #Create Corpus tuple (BoW format) containing the each word id and their frequency


# In[ ]:


#Create dictionary and dataframe of Corpus to remove the high frequncy words 
dict_corpus = {}

for i in range(len(corpus)):
  for idx, freq in corpus[i]:
    if id2word[idx] in dict_corpus:
      dict_corpus[id2word[idx]] += freq
    else:
       dict_corpus[id2word[idx]] = freq
       
dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])


# In[ ]:


dict_df


# In[ ]:


# Plot histogram of word frequency 
plt.figure(figsize=(8,6))
sns.histplot(dict_df['freq'], bins=100);


# In[ ]:


#Top 25 high-frequency words
dict_df2=dict_df.sort_values('freq', ascending=False).head(25) 


# In[ ]:


dict_df2


# In[ ]:


# Filter out high-frequancy words based on the pre-defined threshold
extension = dict_df[dict_df.freq>598438].index.tolist()


# In[ ]:


# List of other non-relevant words identified by inspection that need to be filterd out
unrelevant=['problem','reason','luck','book','purpose','nose','clue','happen','scucceed','reason','thought','oppose','hope','realize','cheer','want','example','laptop','guy','look','need','write', 'walk','good', 'lot', 'people', 'great','please','wikipedia','fun','movie', 'problem', 'answer','thank','need', 'question', 'thing','suggest','solution', 'wrong', 'prefer', 'mention', 'correctly', 'good', 'easy', 'follow', 'great', 'feel', 'idea', 'recommend', 'support','want','look','way','pretty', 'result', 'basic', 'give', 'bad', 'nice', 'try', 'well', 'write', 'look']


# In[ ]:


# Add non-relevant words to high-frequncy words
extension.extend(unrelevant)


# In[ ]:


# Add high-frequency and nor-relevant words to stop words list
stop_words.extend(extension)
# Rerun the word processing function
processed_data= process_words(QAT_list)
# Recreate Dictionary
id2word = corpora.Dictionary(processed_data)
print('New Vocabulary Size:', len(id2word))


# In[ ]:


# Filter out words that occur on less than 10 questions, or on more than 60% of the questions.
id2word.filter_extremes(no_below=10, no_above=0.6)
print('Total Vocabulary Size:', len(id2word))


# In[ ]:


# Update the corpus 
corpus = [id2word.doc2bow(text) for text in processed_data]


# In[ ]:


import pickle
pickle.dump(corpus, open("corpus_100_QAT.pkl", "wb")) #Pickle the corpus and id2word to restart the kernel with fresh memory
pickle.dump(id2word, open("id2word_100_QAT.pkl", "wb"))


# In[ ]:


import os
from gensim.models.wrappers import LdaMallet  # Import Mallet LDA
num_topics=100 # Number of Topics

# Path to Mallet Change to correct path
os.environ['MALLET_HOME'] = '\\Users\\hamed\\mallet-2.0.8' 
mallet_path = '\\Users\\hamed\\mallet-2.0.8\\bin\\mallet'


# In[ ]:


# Built the topic model using Mallet LDA implementation
ldamallet =gensim.models.wrappers.LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)


# In[ ]:


# Print the topics compiled by the model and diplay 5 term and their relative weights
from pprint import pprint
pprint(ldamallet.show_topics(num_words=5,formatted=False))


# In[ ]:


# Compute the Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=processed_data, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('Coherence Score: ', coherence_ldamallet)


# In[ ]:


import pickle
corpus = pickle.load(open("corpus_100_QAT.pkl", "rb"))
id2word = pickle.load(open("id2word_100_QAT.pkl", "rb"))


# In[ ]:


# Build another model using multicore LDA implementation and compare the coherence score
from gensim.models import LdaMulticore
ldamulticore = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=id2word,workers=4, eval_every=None, passes=20, batch=True,per_word_topics=True)


# In[ ]:


# Display topics
from pprint import pprint
pprint(ldamulticore.show_topics(num_words=5,formatted=False))


# In[ ]:


# Compute Coherence Score for the multicore model
processed_data = pickle.load(open("processed_data_100_QAT.pkl", "rb"))
coherence_model_ldamulticore = CoherenceModel(model=ldamulticore, texts=processed_data, dictionary=id2word, coherence='c_v')
coherence_ldamulticore = coherence_model_ldamulticore.get_coherence()
print('Coherence Score: ', coherence_ldamulticore)


# In[ ]:


# Build another model using LDA implementation and compare the coherence score with the two previous models
from gensim.models import LdaModel
ldamodel = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, eval_every=None, passes=20,per_word_topics=True)


# In[ ]:


# Display topics
pprint(ldamodel.show_topics(num_words=5,formatted=False))


# In[ ]:


# Compute Coherence Score
coherence_model_ldamodel = CoherenceModel(model=ldamodel, texts=processed_data, dictionary=id2word, coherence='c_v')
coherence_ldamodel = coherence_model_ldamodel.get_coherence()
print('Coherence Score: ', coherence_ldamodel)


# In[ ]:


def plot_difference_plotly(mdiff, title="", annotation=None):
    """Plot the difference between models using plotly. The chart will be interactive"""
    import plotly.graph_objs as go
    import plotly.offline as py

    annotation_html = None
    if annotation is not None:
        annotation_html = [
            [
                "+++ {}<br>--- {}".format(", ".join(int_tokens), ", ".join(diff_tokens))
                for (int_tokens, diff_tokens) in row
            ]
            for row in annotation
        ]

    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html)
    layout = go.Layout(width=950, height=950, title=title, xaxis=dict(title="topic"), yaxis=dict(title="topic"))
    py.iplot(dict(data=[data], layout=layout))


# In[ ]:


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """ function to plot difference between modelsUses using matplotlib"""
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


# In[ ]:


# Heatmap to compare the correlation between LDA and Multicore LDA
mdiff, annotation = ldamodel.diff(ldamulticore, distance='jaccard', num_words=30)
plot_difference(mdiff, title="LDA vs LDA Multicore Topic difference by Jaccard distance", annotation=annotation)


# In[ ]:


# Convert LdaMallet model to a gensim model for Visualization

from gensim.models.ldamodel import LdaModel

def convertldaMalletToldaGen(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha) 
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim


# In[ ]:


ldagensim = convertldaMalletToldaGen(ldamallet)


# In[ ]:


# Heatmap to compare the correlation between LDA and Multicore LDA
mdiff, annotation = ldagensim.diff(ldamulticore, distance='jaccard', num_words=30)
plot_difference(mdiff, title="LDA Mallet vs LDA Multicore Topic difference by Jaccard distance", annotation=annotation)


# In[ ]:


# Get the topic distributions by passing in the corpus to the model
tm_results = ldamallet[corpus]


# In[ ]:


# Get the most dominant topic
corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]


# In[ ]:


# Get top  significant terms and their probabilities for each topic using ldamallet
topics = [[(term, round(wt, 3)) for term, wt in ldamallet.show_topic(n, topn=20)] for n in range(0, ldamallet.num_topics)]


# In[ ]:


# Get top  significant terms and their probabilities for each topic using LDA multicore
topics_ldamulticore = [[(term, round(wt, 3)) for term, wt in ldamulticore.show_topic(n, topn=20)] for n in range(0, ldamulticore.num_topics)]


# In[ ]:


import pickle
from gensim.models import CoherenceModel
ldamodel = pickle.load(open("\\Users\\hamed\\Desktop\\ECE 143 Project Data Files\\ldamodel_100_QAT.pkl", "rb"))
ldamulticore = pickle.load(open("\\Users\\hamed\\Desktop\\ECE 143 Project Data Files\\ldamulticore_100_QAT.pkl", "rb"))


# In[ ]:


# Get top  significant terms and their probabilities for each topic using LDA multicore
topics_ldam = [[(term, round(wt, 3)) for term, wt in ldamodel.show_topic(n, topn=20)] for n in range(0, ldamodel.num_topics)]


# In[ ]:


# 5 most probable words for each topic for LDA 
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics_ldamulticore], columns = ['Term'+str(i) for i in range(1, 21)], index=['Topic '+str(t) for t in range(1, ldamulticore.num_topics+1)]).T
topics_df.head()


# In[ ]:


# 5 most probable words for each topic for LDA  
topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], columns = ['Term'+str(i) for i in range(1, 21)], index=['Topic '+str(t) for t in range(1, topics.num_topics+1)]).T
topics_df.head()


# In[ ]:


# Display all the terms for each topic for LDA Mallet
#pd.set_option('display.max_colwidth', -1)
#topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, ldamallet.num_topics+1)] )
#topics_df


# In[ ]:


# Display all the terms for each topic for LDA Multicore
pd.set_option('display.max_colwidth', -1)
topics_df_ldam = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics_ldamulticore], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, ldamulticore.num_topics+1)] )
topics_df_ldam


# In[ ]:


# Display all the terms for each topic for LDA 
import pandas as pd
pd.set_option('display.max_colwidth', -1)
topics_df_lda = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics_ldam], columns = ['Terms per Topic'], index=['Topic'+str(t) for t in range(1, ldamodel.num_topics+1)] )
topics_df_lda


# In[ ]:


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


# In[ ]:


# Visualize the LDA Multicore terms as wordclouds
from wordcloud import WordCloud # Import wordclouds

# Initiate the wordcloud object
wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [20, 15]
plt.title("LDA Multicore Output")


# Create subplots for each topic
for i in range(25):
    wc.generate(text=topics_df_ldam["Terms per Topic"][i])
    plt.subplot(5, 5, i+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(topics_df_ldam.index[i])
plt.show()


# In[ ]:


# Visualize the LDA terms as wordclouds
from wordcloud import WordCloud # Import wordclouds

# Initiate the wordcloud object
wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [20, 15]
plt.title("LDA Output")
startoffset = 25
# Create subplots for each topic
for i in range(25):
    wc.generate(text=topics_df_lda["Terms per Topic"][i+startoffset])
    plt.subplot(5, 5, i+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(topics_df_lda.index[i])

plt.show()


# In[ ]:


# Visualize the LDA terms as wordclouds
from wordcloud import WordCloud # Import wordclouds

# Initiate the wordcloud object
wc = WordCloud(background_color="white", colormap="Dark2", max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [10, 5]
plt.title("LDA Output")

# Create subplots for each topic

wc.generate(text=topics_df_ldam["Terms per Topic"][75])
plt.subplot(1, 1, 1)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")

plt.show()


# In[ ]:


import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(ldagensim, corpus, id2word, sort_topics=False)
pyLDAvis.display(vis_data)


# In[ ]:


import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(ldamulticore, corpus, id2word, sort_topics=False)
pyLDAvis.display(vis_data)


# In[ ]:


import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(ldamodel, corpus, id2word, sort_topics=False)
pyLDAvis.display(vis_data)


# In[ ]:


import pickle #picke the result for later use
pickle.dump(processed_data, open("processed_data_100_QAT.pkl", "wb"))
pickle.dump(ldamulticore, open("ldamulticore_100_QAT.pkl", "wb"))
pickle.dump(ldamodel, open("ldamodel_100_QAT.pkl", "wb"))


# In[ ]:


import pickle #Unpickle the previous results
from gensim.models import CoherenceModel
corpus = pickle.load(open("C:\\Users\\hamed\\Desktop\\ECE 143 Project Data Files\\corpus_100_QAT.pkl", "rb"))
id2word = pickle.load(open("C:\\Users\\hamed\\Desktop\\ECE 143 Project Data Files\\id2word_100_QAT.pkl", "rb"))
processed_data = pickle.load(open("C:\\Users\\hamed\\Desktop\\ECE 143 Project Data Files\\processed_data_100_QAT.pkl", "rb"))


# In[ ]:


import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(ldamodel, corpus, id2word, sort_topics=False)
pyLDAvis.display(vis_data)


# In[ ]:


import pyLDAvis #Import pyLDAvis for interactive visualization
pyLDAvis.display(vis_data)


# In[ ]:


import pyLDAvis # Save the interactive 
pyLDAvis.save_html(vis_data, "C:\\Users\\hamed\\Desktop\\ECE 143 Project Data Files\\lda_visulaization_100.html")


# In[ ]:


# Visualization of "Tags" distribution
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import collections
from collections import OrderedDict
file_tags = 'C:\\Users\\hamed\\Desktop\\ECE 143 Project Data Files\\Tags.csv'
Tags_r = pd.read_csv(file_tags, engine ='python',usecols = ['Tag'], encoding = 'iso-8859-1',error_bad_lines=False)
tags_count = collections.Counter(Tags)
common_tags = OrderedDict(tags_count.most_common(18))
del common_tags['python'] # remove non-infomative tags
del common_tags['python-2.7']
del common_tags['python-3.x']
labels, values = zip(*common_tags.items())
values = [v/1000 for v in values] # Change the frequency to thousends
s = pd.Series(
   values, labels
)
plt.figure(figsize=(80,30))

ax = s.plot.bar(x="labels", y="values", rot=0,color=[(174/255,199/255,232/255)])
plt.title('Top 15 Most Frequent Tags \n', fontsize=60,fontfamily='Nunito')
plt.xlabel('\nTags', fontsize=55,fontfamily='Nunito')
plt.ylabel('   Frequency \n (thousands)\n ', fontsize=55,fontfamily='Nunito')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
from matplotlib import rcParams
plt.rcParams.update({'font.size': 40,'font.family':'Nunito'})
rects = ax.patches
# Make some labels.

labels = ['62.8k','26.8k','25.8k','18.9k','16.5k','14k','13.4k','10.7k','10.6k','10.4k','10.2k','9.3k','9.1k','8k','7.5k']

# Add labels on top of the bars
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,
            ha='center', va='bottom')    
plt.show()


# In[ ]:


# Compute coherance score for different topic values
lda_models, coherence_scores = topic_model_coherence_generator(corpus=corpus, texts=processed_data, dictionary=id2word, start_topic_count=25, end_topic_count=150, step=25)


# In[ ]:


import gensim #Import gensim for topic modelling 
import gensim.corpora as corpora
from gensim.models import LdaModel
import os
format =".pkl"

def topic_model_coherence_generator(corpus, texts, dictionary, start_topic_count=25, end_topic_count=125, step=25):
    """Get the coherance score for differnt topic values from 25 and 150"""
  models = []
  coherence_scores = []
  for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
    ldamodel = LdaModel(corpus=corpus, num_topics=topic_nums, id2word=id2word, eval_every=None, passes=20,per_word_topics=True)
    dir_name="C:\\Users\\hamed\\OneDrive\\Desktop\\ECE 143 Project Data Files\\"
    save_dir=os.path.join(dir_name, "ldamodel_"+ str(topic_nums) +"_QAT_3_7" + format)
    pickle.dump(ldamodel, open(save_dir, "wb"))
    
    cv_coherence_model_lda = gensim.models.CoherenceModel (model=ldamodel, corpus=corpus, texts=texts,
                                                                     dictionary=dictionary, coherence='c_v')
      
    coherence_score = cv_coherence_model_lda.get_coherence()
    coherence_scores.append(coherence_score)
    models.append(ldamodel)
  return models, coherence_scores  

