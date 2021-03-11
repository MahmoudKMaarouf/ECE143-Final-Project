from io import StringIO
from html.parser import HTMLParser

import pandas as pd
import gensim
import numpy as np
import scipy
import pickle5 as pickle
from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager


class StackOverflowParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.strict = False  # If any invalid html is encountered, parser will make a best guess at its intention
        self.convert_charrefs = True  # Hold data section until next tag is encountered

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
        assert isinstance(tag, str)
        assert isinstance(attrs, list)

        self.lasttag = tag

    def handle_data(self, data):
        '''
        Method inherited from HTMLParser super class that is called whenever data inside of a tag is encountered.
        In this parser, it saves blocks of code to the field variable self.code and records all text with HTML tags removed
        :param data: Current data inside of a tag being parsed
        :type tag: str
        '''
        assert isinstance(data, str)

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
    assert isinstance(html, str)

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
    assert isinstance(html, str)

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
    assert isinstance(html, str)

    s = StackOverflowParser()
    s.feed(html)
    return [item.replace('\n', ' ') for item in s.code_blocks]


# import pandas as pd

# File paths
file_questions = 'Questions.csv'
file_answers = 'Answers.csv'
file_tags = 'Tags.csv'

dates = ["CreationDate"]



# # Load dataframes (only loading first 10000 rows for now to reduce processing time)
# questions_df = pd.read_csv(file_questions, nrows=1000, encoding='iso-8859-1')
# answers_df = pd.read_csv(file_answers, nrows=1000, encoding='iso-8859-1')
questions_df = pd.read_csv(file_questions, encoding='iso-8859-1')
answers_df = pd.read_csv(file_answers, encoding='iso-8859-1')
# tags_df = pd.read_csv(file_tags, encoding = 'iso-8859-1', nrows=10000)
# In[ ]:


# Add extra columns to dataframes
# This takes a long time (~10 minutes) to process the entire dataset
# Might be worth exploring the pandas to_pickle() method for saving/loading the dataframes
# questions_df['Body_no_tags'] = questions_df['Body'].apply(strip_tags)
questions_df['Body_no_tags_no_code'] = questions_df['Body'].apply(get_text_no_code)
# questions_df['Body_code'] = questions_df['Body'].apply(get_code)

# answers_df['Body_no_tags'] = answers_df['Body'].apply(strip_tags)
answers_df['Body_no_tags_no_code'] = answers_df['Body'].apply(get_text_no_code)
# answers_df['Body_code'] = answers_df['Body'].apply(get_code)


# Row 11 in Questions.csv is a good example with a few code blocks
# body = questions_df['Body'][11]
# body_no_tags = questions_df['Body_no_tags'][11]
# body_code = questions_df['Body_code'][11]
# body_no_tags_no_code = questions_df['Body_no_tags_no_code'][11]
questions = questions_df['Body_no_tags_no_code']
answers = answers_df['Body_no_tags_no_code']




from textblob import TextBlob


def sentiment_analysis(comment):
    '''
    This function calculate the polarities of each strings from the preprocessed questions/answers strings and store
    them into a dictionary.
    :param comment:
    :return:
    '''
    dict_sentiment = {}
    for feedback in range(len(comment)):
        pol = TextBlob(comment[feedback]).sentiment.polarity
        dict_sentiment[feedback] = pol

    return dict_sentiment


question_dict = sentiment_analysis(questions)
answer_dict = sentiment_analysis(answers)


def find_polarities(adict):
    pos_dict = {}
    neu_dict = {}
    neg_dict = {}
    for key, values in adict.items():
        if values > 0:
            pos_dict[key] = values
        elif values == 0:
            neu_dict[key] = values
        else:
            neg_dict[key] = values

    return pos_dict, neu_dict, neg_dict


q_pos, q_neu, q_neg = find_polarities(question_dict)
a_pos, a_neu, a_neg = find_polarities(answer_dict)

def find_general_tone(adict):
    all_val = adict.values()
    max_val = max(all_val)
    min_val = min(all_val)

    avg = sum(adict.values()) / len(adict)

    return max_val, min_val, avg


find_general_tone(question_dict)
def plot_sentiment_overall():
    q_max, q_min, q_avg = find_general_tone(question_dict)
    a_max, a_min, a_avg = find_general_tone(answer_dict)


    w = 0.2

    bar1 = np.arange(1)
    bar2 = bar1 + w
    bar3 = bar2 + w
    font = {'fontname': 'Nunito'}
    plt.subplot(1,2,1)

    # plt.bar(bar, [q_max, q_min, q_avg], w, label='Most Positive Polarity')
    plt.bar(bar1, q_max, w, color=(152/255, 223/255, 138/255), label='Most Positive Polarity')
    plt.bar(bar2, q_min, w, color=(255/255, 152/255, 150/255), label='Most Negative Polarity')
    plt.bar(bar3, q_avg, w, color=(158/255, 218/255, 229/255), label='Average Polarity')

    # plt.rcParams['font.family'] = 'Nunito'
    plt.title('Questions', fontsize=16, **font)
    plt.ylabel('Polarity', fontsize=16, **font)
    plt.xticks([], [])
    plt.legend(prop={'family': 'Nunito', 'size':15})
    plt.rcParams.update({'font.family': 'Nunito'})
    # plt.show()

    plt.subplot(1,2,2)
    plt.bar(bar1, a_max, w, color=(152/255, 223/255, 138/255), label='Most Positive Polarity')
    plt.bar(bar2, a_min, w, color=(255/255, 152/255, 150/255), label='Most Negative Polarity')
    plt.bar(bar3, a_avg, w, color=(158/255, 218/255, 229/255), label='Average Polarity')
    plt.xticks([], [])

    plt.ylabel('Polarity', fontsize=16, **font)
    plt.title('Answers', fontsize=16, **font)
    # plt.bar(bar1, max_values, w, label='Most Positive Polarity')
    # plt.bar(bar2, min_values, w, label='Most Negative Polarity')
    # plt.bar(bar3, avg_values, w, label='Average Polarity')
    # plt.xticks(bar1+w, 'Questions')

    plt.legend(prop={'family': 'Nunito', 'size':15})
    plt.rcParams.update({'font.family': 'Nunito'})
    #
    plt.show()

plot_sentiment_overall()


ldamulticore = pickle.load(open("ldamulticore_100_QA.pkl", "rb"))
# pprint(ldamulticore.show_topics(num_words=5,formatted=False))
topics = [[(term, round(wt, 3)) for term, wt in ldamulticore.show_topic(n, topn=20)] for n in range(0, ldamulticore.num_topics)]

pd.set_option('display.max_colwidth', -1)

topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns=['Terms per Topic'], index=['Topic'+str(t) for t in range(1, ldamulticore.num_topics+1)])

print(topics_df)
topics_split = topics_df['Terms per Topic'].str.split(',')


def getsentiment_topics(x):
    '''
    This function checks if any word in each topic word list also exists in the strings of the questions/answers.
    If so, calculate the polarity of that string and get the average polarity for all the topics.
    The function should return a dictionary where keys are Topic1-100 and values are the average polarities of the
    corresponding topics.
    '''
    topic_dict = {}
    for i in range(len(topics_split)):
        count = 0
        pol = 0
        word_list = [word.strip() for word in topics_split[i]]
        for q in x:
            q_split = q.split()
            check = any(item in q_split for item in word_list)
            if check:
                count += 1
                pol += TextBlob(q).sentiment.polarity
        if count != 0:
            avg_pol = pol / count
            topic_dict['Topic' + str(i+1)] = avg_pol
        else:
            topic_dict['Topic' + str(i+1)] = 0
    return topic_dict


topic_dict_question = getsentiment_topics(questions)
topic_dict_answers = getsentiment_topics(answers)

def merge_dict(x, y):
    keys = x.keys()
    values = zip(x.values(), y.values())
    combined = dict(zip(keys, values))
    avgdict = {}
    for keys, values in combined.items():
        avgdict[keys] = (values[0] + values[1]) / 2

    return avgdict

avg_dict = merge_dict(topic_dict_question, topic_dict_answers)
print(avg_dict)

keys = avg_dict.keys()
values = avg_dict.values()
clrs = ['red' if (x == max(values)) else 'blue' if (x == min(values)) else 'gray' for x in values]
plt.bar(keys, values, color=clrs)
plt.xlabel('Topics')
plt.ylabel('Polarity')
plt.title('Sentiment Analysis for each Topic')
plt.xticks(rotation=90)
plt.show()













# print(q_pos)


# import operator
# qmax = max(a_pos.items(), key=operator.itemgetter(1))[0]
# qmin = min(a_neg.items(), key=operator.itemgetter(1))[0]
#
# print(qmax, qmin)

#
# print(len(q_pos), len(q_neu), len(q_neg))
# print(len(a_pos), len(a_neu), len(a_neg))