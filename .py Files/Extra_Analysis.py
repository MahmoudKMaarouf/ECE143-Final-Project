#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
    if not isinstance(html,str):
        #print(html)
        html = 'NAN' 
    #assert isinstance(html,str)
    
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


# In[3]:


import pandas as pd

# File Paths 
file_questions = 'Questions.csv'
file_answers = 'Answers.csv'
file_tags = 'Tags.csv'

dates = ["CreationDate"]

# Load dataframes (only loading first 10000 rows for now to reduce processing time)
questions_df = pd.read_csv(file_questions, encoding = 'iso-8859-1', nrows=10000, parse_dates=dates)
answers_df = pd.read_csv(file_answers, encoding = 'iso-8859-1', nrows=1000, parse_dates=dates)
tags_df = pd.read_csv(file_tags, encoding = 'iso-8859-1', nrows=1000)


# In[4]:


# Add extra columns to dataframes
# This takes a long time (~10 minutes) to process the entire dataset
# Might be worth exploring the pandas to_pickle() method for saving/loading the dataframes

#questions_df.drop(questions_df.iloc[:, 6:22], inplace = True, axis = 1)   #ignore
questions_df = questions_df.fillna('')                                    #Fill nans with empty string
answers_df = answers_df.fillna('') 

questions_df['Body_no_tags']=questions_df['Body'].apply(strip_tags)
questions_df['Body_no_tags_no_code']=questions_df['Body'].apply(get_text_no_code)
questions_df['Body_code']=questions_df['Body'].apply(get_code)

answers_df['Body_no_tags']=answers_df['Body'].apply(strip_tags)
answers_df['Body_no_tags_no_code']=answers_df['Body'].apply(get_text_no_code)
answers_df['Body_code']=answers_df['Body'].apply(get_code)


# In[5]:


#questions_df = questions_df.drop(questions_df.ix[:, '6':'21'].columns, axis = 1) 
questions_df
#answers_df


# In[6]:


# Creates one database with every question linked to every answer
# For questions with no answer the values are NaN
suffixes = ['.q', '.a']
excess_columns = ['OwnerUserId.q', 'Id.a', 'OwnerUserId.a','ParentId']
QA_df = questions_df.merge(answers_df, how='left', left_on='Id', right_on='ParentId', suffixes=suffixes).drop(excess_columns , axis=1);
QA_df.merge(tags_df, how='left', left_on='Id.q', right_on='Id', suffixes=['', '.t']).drop('Id', axis=1);
QA_df


# In[7]:


# Creates one database with every question linked to every answer
# Questions with no answers are dropped
QA_df = questions_df.merge(answers_df, how='right', left_on='Id', right_on='ParentId', suffixes=suffixes).drop(excess_columns , axis=1);
QAT_df = QA_df.merge(tags_df, how='left', left_on='Id.q', right_on='Id', suffixes=['', '.t']).drop('Id', axis=1);
QAT_df;


# In[8]:


# Row 11 in Questions.csv is a good example with a few code blocks
body = questions_df['Body'][11]
body_no_tags = questions_df['Body_no_tags'][11]
body_code = questions_df['Body_code'][11]
body_no_tags_no_code = questions_df['Body_no_tags_no_code'][11]


# In[9]:


print(body)
print(questions_df)
print(body_no_tags_no_code)
print(body_code)


# In[10]:


#Q_df = QA_df.drop_duplicates(subset=['Id.q'], keep='first')
Q_df = questions_df
Q_df


# In[11]:


import itertools

def getWordsAfter(phrase, string):                                      #String is input text being searched on
    '''
    This function parses the input string and gets the next 
    word of every occurrence of the input "phrase"
    Words after "phrase" seperated by commas are also included
    '''
    assert isinstance(phrase,str)
    assert isinstance(string,str)
    words = []
    pos = 0
    while(string.find(phrase,pos) != -1):      
        index = string.find(phrase, pos)                                 #Gets position of phrase
        pos = pos + index + len(phrase) +1                               #Gets next position
        length = 20 if len(string[pos::]) >= 20 else len(string[pos::])  #Ensures no overflow
        word = string[pos:pos+length]
        tokenized = word.split(",")                                      #Split by comma if it has comma
        for word in tokenized:
            ind = 0
            word = word.lstrip()                                         #Strip whitespace
            for char in word:
                if not char.isalpha():                                   #Must only contain letters
                    break
                ind+=1
            word = word[:ind]
            if word:                       
                words.append(word.lower())                               #Append if valid word
        index+=1
    words = list(dict.fromkeys(words))                                   #Removes duplicates (+ Efficiency)
    return words

def getMostCommonLibraries(body_text):
    '''
    This function will search for the most common libraries by 
    looking at keywords after "import" and "from"
    '''
                                                                         #Removes common words
    wordsToRemove = ["the", "that", "a", "d", "it", "i", "python", "e", "an", 
                     "and", "in", "foo", "t", "r", "to", "this", "some", "s",
                     "s", "n", "but", "within", "self", "m", "my", "b", "o",
                     "er", "c", "if", "y", "import", "on", "remove",
                     "print", "f", "from", "def", "l", "w", "module", "p", 
                     "win", "x", "for", "here", "g", "name", "is", "or", "data",
                     "return", "one", "another", "all", "h", "user" "include", "which", "me",
                     "u", "get", "as", "ort", "he", "w"]
    index = 0
    libraries = []
    for i in range(len(body_text)):
        body = body_text[i]
        import_words = getWordsAfter('import',body)                       #Collects words after import
        if import_words:
            libraries.append(import_words)
            from_words = getWordsAfter('from',body)  #if there is a "from", there is also an "import"
            if from_words:
                libraries.append(from_words)  
    lib_flat = list(itertools.chain.from_iterable(libraries))             #Flattens list
    lib_flat =[word for word in lib_flat if word not in wordsToRemove]    #Removes words from wordsToRemove
    return lib_flat

#print(getMostCommonLibraries(Q_df['Body_no_tags']))                       #Ignores HTML tags when parsing, keeps code blocks


# In[12]:


from collections import Counter

def getMostCommonElementsInList(aList, numElements = 30):              #numElements to adjust max output
    '''
    "This function will get the most common elements in the
    list by summing up their respective number of occurrences
    '''
    assert isinstance(aList,list)
    assert isinstance(numElements,int)
    c = Counter(aList)
    return c.most_common(numElements)
    
libraries = getMostCommonLibraries(Q_df['Body_no_tags'])
common = getMostCommonElementsInList(libraries,50)
#print(common)


# In[13]:


def getMostCommonCategory(titles, body_text, categories):
    '''
    This function is able to search the title and body of a 
    StackOverflow question for certain categories.
    Categories is a dictionary that will increment the key
    of a count dictionary if its corresponding data members
    are in the title or body of the post.
    '''
    assert isinstance(titles,pd.Series) or isinstance(titles,list)
    assert isinstance(body_text,pd.Series) or isinstance(body_text,list)
    assert isinstance(categories,dict)
    categoryCount = dict()
    for key in categories.keys():
        categoryCount[key] = 0
    for i in range(len(body_text)):
        title = titles[i].lower()
        body = body_text[i].lower()
        for key in categories.keys():
            for phrase in categories[key]:
                if title.find(phrase) != -1 or body.find(phrase) != -1:
                    categoryCount[key]+=1
                    break
    return categoryCount


# In[14]:


def getMostCommonOS(titles, body_text):
    '''
    This function calls getMostCommonCategory()
    for a category correspondng to operating systems
    '''
    assert isinstance(titles,pd.Series) or isinstance(titles,list)
    assert isinstance(body_text,pd.Series) or isinstance(body_text,list)
    os = dict()
    os["windows"] = ["windows", "microsoft"]
    os["mac"] = ["mac", "macos", "macintosh"]
    os["linux"] = ["linux", "unix", "ubuntu", "debian", "gentoo", "rhel", "centos", "fedora", "kali", "arch", "kubuntu", "deepin"]
    os["chrome"] = ["chromium", "chrome OS", "chromeos"]
    categoryCount = getMostCommonCategory(titles,body_text, os)
    return categoryCount
def getMostCommonIDE(titles, body_text):
    '''
    This function calls getMostCommonCategory()
    for a category correspondng to python IDEs
    '''
    assert isinstance(titles,pd.Series) or isinstance(titles,list)
    assert isinstance(body_text,pd.Series) or isinstance(body_text,list)
    ide = dict()
    ide["jupyter"] = ["jupyter"]
    ide["pycharm"] = ["pycharm"]
    ide["spyder"] = ["spyder"]
    ide["vscode"] = ["visual studio", "visualstudio", "vscode", "vs code"] #Also includes visual studio
    ide["sublime"] = ["sublime"]
    ide["atom"] = ["atom"]
    ide["vim"] = ["vim"]
    ide["eclipse"] = ["eclipse"]
    ide["emacs"] = ["emacs"]
    ide["gedit"] = ["gedit"]
    ide["rodeo"] = ["rodeo"]
    ide["notepad++"] = ["notepad++"]
    ide["intellij"] = ["intellij"]
    ide["xcode"] = ["xcode"]
    ide["phpstorm"] = ["phpstorm"]
    ide["netbeans"] = ["netbeans"]
    categoryCount = getMostCommonCategory(titles, body_text, ide)
    return categoryCount
def getMostCommonPM(titles, body_text): #PM = Package Manager 
    '''
    This function calls getMostCommonCategory()
    for a category correspondng to python PMs
    '''
    assert isinstance(titles,pd.Series) or isinstance(titles,list)
    assert isinstance(body_text,pd.Series) or isinstance(body_text,list)
    pm = dict() 
    pm["pip"] = ["pip"]
    pm["conda"] = ["conda"] 
    categoryCount = getMostCommonCategory(titles,body_text, pm)
    return categoryCount

os = getMostCommonOS(Q_df['Title'], Q_df['Body_no_tags'])
for key in os.keys():
    print(key + ":" , os[key])
print("\n")
ide = getMostCommonIDE(Q_df['Title'], Q_df['Body_no_tags'])
for key in ide.keys():
    print(key + ":" , ide[key])
print("\n")
pm = getMostCommonPM(Q_df['Title'], Q_df['Body_no_tags'])
for key in pm.keys():
    print(key + ":" , pm[key])


# In[15]:


from collections import OrderedDict 

def sortQuestionsByYear(df_title, df_body, df_qdates):
    '''
    This function will return 2 dictionaries, one for the titles
    and one for the body of the questions, with each key corresponding
    to the month-yr of the respective category. 
    '''
    assert isinstance(df_title,pd.Series)
    assert isinstance(df_body,pd.Series)
    assert isinstance(df_qdates,pd.Series)
    titles = OrderedDict()
    questions = OrderedDict()
    index = 0
    for q in df_qdates:
        year = str(q)[:7]
        if year.find('-') != 4: 
            continue                 #Avoids corrupt data                   
        if year not in titles.keys():
            titles[year] = []
        if year not in questions.keys():
            questions[year] = []
        titles[year].append(df_title[index])
        questions[year].append(df_body[index])
        index+=1
    return titles, questions

titlesByYear, questionsByYear = sortQuestionsByYear(Q_df['Title'],Q_df['Body_no_tags'],Q_df['CreationDate'])
#print (list(questionsByYear.items())[0])        #Gets first year-month pair 2008-08
#print (list(questionsByYear.keys()))        #Gets first year-month pair 2008-08


# In[16]:


def getSortedData(func, titlesByYear, questionsByYear, getMostCommon = False):
    '''
    Returns the mappings of data pairs based on their category, and does this for
    each year-month pair
    '''
    isFunc = False
    if callable(func): isFunc  = True
    assert isFunc ==True
    assert isinstance(titlesByYear,OrderedDict)
    assert isinstance(questionsByYear,OrderedDict)
    assert isinstance(getMostCommon, bool)
    
    data_dict = OrderedDict()
    for year in questionsByYear.keys():
        if getMostCommon:
            data_dict[year] = getMostCommonElementsInList(func(questionsByYear[year]),15)
        else:
            data_dict[year] = func(titlesByYear[year],questionsByYear[year])
    return data_dict

LibrariesByYear = getSortedData(getMostCommonLibraries, titlesByYear,questionsByYear, True)
OSByYear = getSortedData(getMostCommonOS,titlesByYear,questionsByYear)
IDEByYear = getSortedData(getMostCommonIDE,titlesByYear,questionsByYear)
PMByYear = getSortedData(getMostCommonPM,titlesByYear,questionsByYear)


# In[17]:


print(PMByYear)     #Gets PM data by Year.


# In[18]:


def collectAnimatedPlotData(data_dict, questionsByYear):
    '''
    Puts dictionary into a format readable for animated bar/pie plots
    '''
    assert isinstance(data_dict,OrderedDict)
    assert isinstance(questionsByYear,OrderedDict)
    chartInput = OrderedDict()
    for year in questionsByYear.keys():
        inputList = []
        #print("Year: ",year)
        for key in data_dict[year].keys():
            inputList.append(data_dict[year][key])
            #print(key + ":" , commonOSByYear[year][key])  
        #print("\n") 
        chartInput[year] = inputList
    return chartInput
        
def getCustomLibraryDataLabels(LibrariesByYear, questionsByYear):
    '''
    For every year-month pair, collect the top 15 data, and 
    with its corresponding labels in a separate list
    '''
    assert isinstance(LibrariesByYear,OrderedDict)
    assert isinstance(questionsByYear,OrderedDict)
    libraryBarInput = OrderedDict()
    libraryLabelsList = []
    for year in questionsByYear.keys():
        tempLabels,tempValues = [], []
        for pair in LibrariesByYear[year]:
            tempLabels.append(pair[0])
            tempValues.append(pair[1])
        libraryLabelsList.append(tempLabels)
        libraryBarInput[year] = tempValues
    return libraryLabelsList, libraryBarInput
    
OSPieInput = collectAnimatedPlotData(OSByYear, questionsByYear)
IDEPieInput = collectAnimatedPlotData(IDEByYear, questionsByYear)
PMPieInput = collectAnimatedPlotData(PMByYear, questionsByYear)
libraryLabelsList, libraryBarInput = getCustomLibraryDataLabels(LibrariesByYear, questionsByYear)


# In[19]:


OSLabels = ['Windows', 'Mac', 'Linux', 'ChromeOS']
IDELabels = ["jupyter" "pycharm" "spyder" "vscode" "sublime" "atom", "vim", 
             "eclipse", "emacs", "gedit", "rodeo", "notepad++", "intellij",
             "xcode", "phpstorm", "netbeans"]
PMLabels = ["Pip","Conda"]

print(OSPieInput)    
#print(IDEPieInput)
#print(PMPieInput)
#print(libraryLabelsList)
#print(libraryBarInput)


# In[20]:


# Color Initilaizations for Charts
tableau10light = [(174/255, 199/255, 232/255),(255/255, 187/255, 120/255), (255/255, 152/255, 150/255), (152/255, 223/255, 138/255),(247/255, 182/255, 210/255), (219/255, 219/255, 141/255), (199/255, 199/255, 199/255), (196/255, 156/255, 148/255), (158/255, 218/255, 229/255), (197/255, 176/255, 213/255)]
font = {'fontname':'Nunito'}
blueColor = (174/255, 199/255, 232/255)
grayColor = (197/255, 176/255, 213/255)


# In[24]:


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def createAnimatedPieChart(labels, data):
    '''
    Creates an animated pie chart and saves it as a gif. 
    Data is plotted in the order of the OrderedDict of data
    '''
    explode=[.01]*len(labels)
    nums =[0]*len(labels)
    fig, ax = plt.subplots()
    
    def update(num):
        ax.clear()
        ax.axis('equal')
        nums = list(data.items())[::6][num][1]
        ax.pie(nums, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=140, colors = tableau10light)
        ax.set_title('Year: ' + str((list(data.keys())[::6][num])),**font)            #Gets corresponding year-month 
        plt.rcParams.update({'font.family':'Nunito'})
    anim = FuncAnimation(fig, update, frames=range(len(data)), repeat=False)  #Calls update for next iteration
    #anim.save('./OSpie.gif', writer='imagemagick', fps=60)     #Save as gif
    plt.show()
    
def createAnimatedBarPlot(labels, data):
    '''
    Creates an animated bar plot and saves it as a gif. 
    Data is plotted in the order of the OrderedDict of data
    '''
    temp = OrderedDict()
    
    for i in range(len(data)):
        if i%6==0:
            temp[list(data.keys())[i]] = list(data.values())[i]
    data = temp
    #labels = labels[::6]
    fig, ax = plt.subplots()
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(wi+1.5, hi)
    maxValue = 0
    for key in data.keys():
        if maxValue < max(data[key]):
            maxValue = max(data[key])
    def animate(i):
        x = labels
        if(isinstance(labels[0],list)):
            x = labels[i]
        ax.clear()
        ax.set_title(str((list(data.keys())[i])), **font)    
        plt.xlabel("Count", **font)    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        width = list(data.items())[i][1]
        df = pd.DataFrame({"Labels":x, "Data":width})  
        df_sorted= df.sort_values('Data')           #Using pandas, sort data allowing highest popularity on top
        plt.barh('Labels', 'Data', data=df_sorted, color = blueColor)
        #plt.tight_layout()
        for i, v in enumerate(list(df_sorted['Data'])):
            plt.text(v + 1, i-.05, str(v), color='black', va='center')  #Allows display of bar plot width
        plt.xlim(0, 1.1*maxValue)                 #Preset x coordinate 
        plt.rcParams.update({'font.family':'Nunito'})
    anim= FuncAnimation(fig,animate, repeat = False, frames=len(data)) #Calls update for next iteration
    #anim.save('./IDEbar.gif', writer='imagemagick', fps=60) #Save as gif
    plt.show()


# In[28]:


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def createAnimatedPieChart(labels, data):
    '''
    Creates an animated pie chart and saves it as a gif. 
    Data is plotted in the order of the OrderedDict of data
    '''
    assert isinstance(labels,list)
    assert isinstance(data,OrderedDict)
    explode=[.01]*len(labels)
    nums =[0]*len(labels)
    fig, ax = plt.subplots()
    def update(num):
        ax.clear()
        ax.axis('equal')
        nums = list(data.items())[num][1]
        ax.pie(nums, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=140)
        ax.set_title('Year: ' + str((list(data.keys())[num])))            #Gets corresponding year-month 
    anim = FuncAnimation(fig, update, frames=range(len(data)), repeat=False)  #Calls update for next iteration
    anim.save('./PMpie.gif', writer='imagemagick', fps=60)     #Save as gif
    #plt.show()
    
def createAnimatedBarPlot(labels, data):
    '''
    Creates an animated bar plot and saves it as a gif. 
    Data is plotted in the order of the OrderedDict of data
    '''
    assert isinstance(labels,list)
    assert isinstance(data,OrderedDict)
    fig, ax = plt.subplots()
    wi, hi = fig.get_size_inches()
    fig.set_size_inches(wi+1.5, hi)
    maxValue = 0
    for key in data.keys():
        if maxValue < max(data[key]):
            maxValue = max(data[key])
    def animate(i):
        x = labels
        if(isinstance(labels[0],list)):
            x = labels[i]
        ax.clear()
        ax.set_title(str((list(data.keys())[i])))        
        width = list(data.items())[i][1]
        df = pd.DataFrame({"Labels":x, "Data":width})  
        df_sorted= df.sort_values('Data')            #Using pandas, sort data allowing highest popularity on top
        plt.barh('Labels', 'Data', data=df_sorted)
        #plt.tight_layout()
        for i, v in enumerate(list(df_sorted['Data'])):
            plt.text(v + 1, i-.05, str(v), color='black', va='center')  #Allows display of bar plot width
        plt.xlim(0, 1.1*maxValue)                 #Preset x coordinate 
    anim= FuncAnimation(fig,animate, repeat = False, frames=len(data)) #Calls update for next iteration
    anim.save('./IDEbar.gif', writer='imagemagick', fps=60) #Save as gif
    #plt.show()
    


# In[25]:


OSLabels = ['Windows', 'Mac', 'Linux', 'ChromeOS']
IDELabels = ["Jupyter", "PyCharm",  "Spyder", "VS Code", "Sublime", "Atom", "Vim", 
             "Eclipse", "Emacs", "Gedit", "Rodeo", "Notepad++", "IntelliJ",
             "Xcode", "PHPStorm", "NetBeans"]
PMLabels = ["Pip","Conda"]

createAnimatedPieChart(OSLabels,OSPieInput)       #Animations do NOT work in Jupyter Notebook, please run in a .py file..
createAnimatedBarPlot(IDELabels,IDEPieInput)
createAnimatedPieChart(PMLabels,PMPieInput)
createAnimatedBarPlot(libraryLabelsList,libraryBarInput)


# In[26]:


import numpy as np

def getCoordinatesFromDict(data_labels,data_dict):
    '''
    Gets coordinates for plotting using animated data
    '''
    plotData = []
    for i in range(len(data_labels)):
        iter = 0
        points = []
        for key in data_dict.keys():
            if (iter > 96):             #Blocks off incomplete month due to end of data
                break
            points.append([iter,data_dict[key][i]])
            iter +=1
        plotData.append(points)
    return plotData


def plotLineGraph(data_list, labels, title, keep = [], custom = False):
    '''
    Plots line graph based on given coordinates and writes string x labels
    for corresponding year
    '''
    fig, ax = plt.subplots()
    plt.title(title, **font)
    for i in range(len(data_list)):
        if custom:
            if labels[i] not in keep:
                continue
        data = np.array(data_list[i])
        x,y = data.T 
        x_labels = ['']*len(x)
        if len(x)>= 5:  x_labels[5]  = '2009'
        if len(x)>= 17: x_labels[17] = '2010'
        if len(x)>= 29: x_labels[29] = '2011'
        if len(x)>= 41: x_labels[41] = '2012'
        if len(x)>= 53: x_labels[53] = '2013'
        if len(x)>= 65: x_labels[65] = '2014'
        if len(x)>= 77: x_labels[77] = '2015'
        if len(x)>= 89: x_labels[89] = '2016'
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel("Year",**font)
        plt.ylabel("Count",**font)
        plt.xticks(range(len(x_labels)), x_labels, size='small',**font)  #Labels corresponding to years
        plt.plot(x,y, label= labels[i],color = tableau10light[i])   #Shows labels of line
        plt.legend()   #Adds legend
        plt.rcParams.update({'font.family':'Nunito'})
    plt.show()


# In[30]:


import numpy as np

def getCoordinatesFromDict(data_labels,data_dict):
    '''
    Gets coordinates for plotting using animated data
    '''
    assert isinstance(data_labels,list)
    assert isinstance(data_dict,OrderedDict)
    plotData = []
    for i in range(len(data_labels)):
        iter = 0
        points = []
        for key in data_dict.keys():
            if (iter > 96):             #Blocks off incomplete month due to end of data
                break
            points.append([iter,data_dict[key][i]])
            iter +=1
        plotData.append(points)
    return plotData


def plotLineGraph(data_list, labels, title, keep = [], custom = False):
    '''
    Plots line graph based on given coordinates and writes string x labels
    for corresponding year
    '''
    assert isinstance(data_list,list)
    assert isinstance(labels,list)
    assert isinstance(title,str)
    assert isinstance(custom,bool)
    plt.title(title)
    for i in range(len(data_list)):
        if custom:
            if labels[i] not in keep:
                continue
        data = np.array(data_list[i])
        x,y = data.T 
        x_labels = ['']*len(x)   #Label indexes dependent on data length...
        if len(x)>= 5:  x_labels[5]  = '2009'
        if len(x)>= 17: x_labels[17] = '2010'
        if len(x)>= 29: x_labels[29] = '2011'
        if len(x)>= 41: x_labels[41] = '2012'
        if len(x)>= 53: x_labels[53] = '2013'
        if len(x)>= 65: x_labels[65] = '2014'
        if len(x)>= 77: x_labels[77] = '2015'
        if len(x)>= 89: x_labels[89] = '2016'
        plt.xticks(range(len(x_labels)), x_labels, size='small')  #Labels corresponding to years
        plt.plot(x,y, label= labels[i])   #Shows labels of line
        plt.legend()   #Adds legend
    plt.show()


# In[27]:


print(OSPieInput)

OSplotData = getCoordinatesFromDict(OSLabels,OSPieInput)
plotLineGraph(OSplotData, OSLabels, "OS Appearances Over Time")

IDEplotData = getCoordinatesFromDict(IDELabels,IDEPieInput)
plotLineGraph(IDEplotData, IDELabels, "IDE Appearances Over Time", ["PyCharm", "Jupyter", "Eclipse", "VS Code"], True)

PMplotData = getCoordinatesFromDict(PMLabels,PMPieInput)
plotLineGraph(PMplotData, PMLabels, "PM Appearances Over Time")


# In[28]:


def reformatLibraryDict(libraryLabelsList, libraryBarInput, newLabels):
    '''
    Format most common library data for plotting 
    (Stored differently due to different algorithm used)
    '''
    libraryDict = OrderedDict()
    LibPlotLabels = []
    LibPlotData = []
    for i in range(len(libraryLabelsList)):
        index = 0
        added = []
        for label in libraryLabelsList[i]:         #If the input newLabels match any of the top 15 labels per given year-month
            for newlabel in newLabels:
                if newlabel == label:
                    if newlabel not in libraryDict.keys():
                        libraryDict[newlabel] = []
                    libraryDict[newlabel].append(list(libraryBarInput.items())[i][1][index])
                    added.append(newlabel)
            index +=1
        for newlabel in newLabels:                #Appends last value into plot data for newLabel if not present in labels
            if newlabel not in added:
                if newlabel not in libraryDict.keys():
                        libraryDict[newlabel] = [0]
                else:
                    libraryDict[newlabel].append(libraryDict[newlabel][-1])
    for key in libraryDict.keys():
        LibPlotLabels.append(key)
        temp = []
        index = 0
        for data in libraryDict[key]:
            if (index > 96):             #Blocks off incomplete month due to end of data
                break
            temp.append([index, libraryDict[key][index]])   #Formats data for plotting
            index +=1 
        LibPlotData.append(temp)
    return LibPlotLabels, LibPlotData


# In[29]:


LibNewLabels = ["numpy", "sys", "os", "django", "pandas", "nt", "matplotlib"]
LibPlotLabels, LibPlotData = reformatLibraryDict(libraryLabelsList, libraryBarInput, LibNewLabels)
plotLineGraph(LibPlotData, LibPlotLabels, "Lib Appearances Over Time")


# In[33]:


def plotTopicSummaryChart():
        '''
        This function will output a donut chart with percentages indiciating how frequent a topic was of a certain
        topic based on the LDA algorithm. 
        '''
        font = {'fontname':'Nunito'}
        colors = [(174/255, 199/255, 232/255),(255/255, 187/255, 120/255), (255/255, 152/255, 150/255),(247/255, 182/255, 210/255), (219/255, 219/255, 141/255),(152/255, 223/255, 138/255), (199/255, 199/255, 199/255), (196/255, 156/255, 148/255), (158/255, 218/255, 229/255), (197/255, 176/255, 213/255), (255/255, 200/255, 186/255)]
        blueColor = (174/255, 199/255, 232/255)
        grayColor = (197/255, 176/255, 213/255)
        topics = ["Basics", "Functions/Classes/OOP", "Specific Modules/Libraries", "File I/O", 
                "Data Manipulation/Handling" , "Web Applications/Frameworks", "Other", "Data Visualization",
                "Setup", "GUI Programming", "Error/Exception Handling"]
        values = [13,12,12,10,9,8,6,5,3,3,3]
        fig, ax = plt.subplots(figsize=(24, 12), subplot_kw=dict(aspect="equal"))
        ax.set_title("Topic Frequency in Stack Overflow", **font)
        wedges, texts, percent = ax.pie(values,  autopct='%1.1f%%', pctdistance=.4, wedgeprops=dict(width=0.5), startangle=40, colors = colors)
        plt.rcParams.update({'font.family':'Nunito'})
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")
        for i, p in enumerate(wedges):  
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                ax.annotate(topics[i], xy=(x, y), xytext=(1.2*np.sign(x), 1.1*y),
                                horizontalalignment=horizontalalignment, **kw, **font)
        plt.show()


# In[34]:


plotTopicSummaryChart() #Plots LDA summary

