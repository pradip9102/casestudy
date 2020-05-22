#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;font-size:30px;" > Kids Measure Up! Learning Analysis </h1>

# <img src='kids_measure.jpg'/>

# # List of Contents :
# * Description
# * Problem Statement
# * Source of Data
# * Business Objectives and Constrains
# * Data Overview
# * Features in the Dataset
# * Mapping Real World Problem to Machine Learning Problem
# * Performance Metric

# # Description

# Early learning apps are now becoming very popular for kids (age 3-5 yrs) to learn the basic concepts like length, height , weight and colors etc. PCB Kids Measure app is one of the most popular learning apps across the United-State which is funded by the U.S. department of education. This App allows kids to learn in a fun way by playing games and watching video and can see their most favourite cartoon so they love to come again and again.
# 
# 
# <br>Data Science Bowl is the world's largest Data Science competition and announces a challenging task each year allowing competitors to solve problems for social good. PCB Kids Measure app is their fifth competition which is presented by Booz Allen Hamilton and Kaggle.
# 
# 
# https://www.kaggle.com/c/data-science-bowl-2019/overview

# # problem Statement
# The task is to predict the number of attempts that a child will take to pass the given Assessment using gameplay data.

# # Source of data

# Data set is available in the form of four csv files which are train.csv, test.csv, specs.csv and train_labels.csv.
# 
# Refer : https://www.kaggle.com/c/data-science-bowl-2019/data

# # Business Objectives and Constrain
# ### latency Requirement
# No strict latency constrain but prediction within an hour is preferable.
# ### Effect of misclassification
# Misclassification will couse bad customer experience and it might recude number of installation_id in future.
# ### Performance Objectives
# Predict the data with correct class as many as possible with high precision and high recall.

# # Data Overview

# Data is available in the form of four csv files and brief desciption is below :
# ### train.csv
# <pre>
# <b>File size</b> - 3.16 GB<br />
# <b>Number of entry</b> - 11.3 M<br />
# <b>Number of Columns</b> - 11 <br />
# <b>Name of Columns</b> - {'event_id','game_session','time_stamp','event_data','installation_id','event_count','event_code','game_time','title','type','world'}<br />
# <b>Number of unique installation_id</b> - 17000 <br />
# </pre>
# 
# ### test.csv
# <pre>
# <b>File size</b> - 379.87 MB<br />
# <b>Number of entry</b> - 1.1 M<br />
# <b>Number of Columns</b> - 11 <br />
# <b>Name of Columns</b> - {'event_id','game_session','time_stamp','event_data','installation_id','event_count','event_code','game_time','title','type','world'}<br />
# <b>Number of unique installation_id</b> - 1000 <br />
# </pre>
# 
# ### specs.csv
# <pre>
# <b>File size</b> - 399.29 KB <br />
# <b>Number of entry</b> - 386 <br />
# <b>Number of Columns</b> - 3 <br />
# <b>Name of Columns</b> - {'event_id','info','arg'}<br />
# </pre>
# 
# ### train_labels.csv
# <pre>
# <b>File size</b> - 1.01 MB <br />
# <b>Number of entry</b> - 17690 <br />
# <b>Number of Columns</b> - 7 <br />
# <b>Name of Columns</b> - {'game_session','installation_id','title','num_correct','num_incorrect','accuracy','accuracy_group'}<br />
# </pre>

# # Features in the data set
# This reference blog helps me alot to understand the features of the data set.
# 
# <u>https://medium.com/@boxinthemiddle/pbs-kids-measure-up-learning-analytics-part-1-9facbdafcdb5</u>
# 
# ### train.csv and test.csv

# <table border="1">
# 	<tr>
# 		<th style="text-align:center">Field Name</th>
# 		<th style="text-align:center">Description</th>
# 	</tr>
#     <tr>
#         <td style="text-align:left">event_id</td>
#         <td style="text-align:left">Randomly generated unique identifier for the event type. Maps to event_id
#                                     column in specs table. Total number of unique event _id is 384.
#         </td>
#     </tr>
#     <tr>
#         <td style="text-align:left">game_session</td>
#         <td style="text-align:left">game_session is randomly generated id for specific installation_id.
#                                     Game_session will not change until the user regines And when he/she comes to                                       play next time a new session will be allotted to them.
#         </td>
#     </tr>
#     <tr>
#         <td style="text-align:left">time_stamp</td>
#         <td style="text-align:left">it is the record of the actual time when the game was playing and 
#                                     the time is in the format of yyyy-mm-ddThh:mm:ss.mmmZ.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">event_data</td>
#         <td style="text-align:left">Semi-structured JSON formatted string containing the events parameters.
#                                     Default fields are: event_count</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">installation_id</td>
#         <td style="text-align:left">Unique for each installation and it represent a user.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">event_count</td>
#         <td style="text-align:left">Counts of number of games played after starting game_session</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">event_code</td>
#         <td style="text-align:left"> Identifier of the event 'class'. Unique per game, but may be duplicated across                                      games. E.g. event code '2000' always identifies the 'Start Game' event for all                                      games. Extracted from event_data.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">game_time</td>
#         <td style="text-align:left">Time in milliseconds since the start of the game session. Extracted from                                           event_data.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">title</td>
#         <td style="text-align:left">Title of the game or video.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">type</td>
#         <td style="text-align:left">Media type of the game or video. Possible values are:
#             <ol>
#                 <li> <strong>Game</strong> - game is task based . In order to go on to the next level a user has to                                          <br>finish the task for that level. </li>
#                 <li> <strong>Clip</strong> - user will finish the task where concept or rules are explained in a                                            <br>short video. </li>
#                 <li> <strong>Activity</strong> - This is comparatively long where children do not have any specific                                          <br>task . They can finish the activity when they are done. </li>
#                 <li> <strong>Assessment</strong> - The user will reach an assessment after finishing several games                                          <br>, clips and activity. Assessment can be thought of as a fun exam where                                        <br>user will use all the skills which they have learnt in previous.</li>
#            </ol>
#         </td>
#     </tr> 
#     <tr>
#         <td style="text-align:left">worls</td>
#         <td style="text-align:left">Specific set of measurement.
#            <ol>
#                 <li> <strong>None</strong> - At the apps start screen </li>
#                 <li> <strong>TREETOPCITY</strong> - for height and length retaled skills. </li>
#                 <li> <strong>MAGMAPEAK</strong> - for capacity related skills. </li>
#                 <li> <strong>CRYSTALCAVES</strong> - for weights related skills</li>
#            </ol>
#         </td>
#     </tr>
# </table>

# ### specs.csv
# This file contains specification of various event types.

# <table border='1'>
#     <tr>
#         <th style="text-align:center">Field Name</th>
#         <th style="text-align:center">Description</th>
#     </tr>
#     <tr>
#         <td style="text-align:left">event_id</td>
#         <td style="text-align:left">Global unique identifier for the event type. Joins to event_id column in events                                     table.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">info</td>
#         <td style="text-align:left">Description of the event</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">args</td>
#         <td style="text-align:left">JSON formatted string of event arguments. Each argument contains:
#         <ol>
#             <li><strong>name</strong> - Argument name.</li>
#             <li><strong>type</strong> - Type of the argument (string, int, number, object, array).</li>
#             <li><strong>info</strong> - Description of the argument.</li>
#         </ol>
#         </td>
#     </tr>
#     
# </table>

# ### train_labels.csv
# The file train_labels.csv has been provided to show how these groups would be computed on the assessments in the training set. Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110. If the attempt was correct, it contains "correct":true.

# <table border='1'>
#     <tr>
#         <th style="text-align:center">Field Name</th>
#         <th style="text-align: center">Description</th>
#     </tr>
#     <tr>
#         <td style="text-align:left"> game_session</td>
#         <td style="text-align:left">game_session is randomly generated id for specific installation_id.
#             Game_session will not change until the user regines
#             And when he/she comes to play next time a new session will be allotted to them.
#         </td>
#     </tr>
#     <tr>
#         <td style="text-align:left">installation_id</td>
#         <td style="text-align:left">Unique for each installation and it represent a user.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">title</td>
#         <td style="text-align:left">Title of the game or video.</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">num_correct</td>
#         <td style="text-align:left">number of True(correct) counts for an Assessment event having event_code = 4100 
#                 and 4110. Only bird measurer Assessment has event_code 4110</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">num_incorrect</td>
#         <td style="text-align:left">number of False(incorrect) counts for an Assessment event having event_code =                                       4100 and 4110. Only bird measurer Assessment has event_code 4110</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">accuracy</td>
#         <td style="text-align:left">it is ratio of num_correct and (num_correct + num_incorrect)</td>
#     </tr>
#     <tr>
#         <td style="text-align:left">accuracy_group</td>
#         <td style="text-align:left">The outcomes in this competition are grouped into 4 groups 
#             <br>(labeled accuracy_group in the data):
#             <ol>
#                 <li><strong>3</strong>: the assessment was solved on the first attempt</li>
#                 <li><strong>2</strong>: the assessment was solved on the second attempt</li>
#                 <li><strong>1</strong>: the assessment was solved after 3 or more attempts</li>
#                 <li><strong>0</strong>: the assessment was never solved</li>
#             </ol>
#         </td>
#     </tr>
# </table>

# # Mapping Real world problem to Machine Learning problem

# ### Type of Machine Learning Problem
# This is multiclass classification task. For given query data we need to predict the number of attempts that the user will take to pass the assessment and decide accuracy_group accordingly. 

# # Performance Metrics (Quadratic Weighted Kappa (QWK))

# Final score is evaluated using quadratic weighted kappa (QWK) which is measure of agreement between actual and predicted class. The range of this metric is generally 0 to 1. If QWK is 1 means actual and predicted class matches perfectly and if QWK is 0 means agreement happened by chance. QWK value can also take negative value which means the prediction is worse than by chance. QWK is calculated by 
# 
# Number of classes is $N=4$ in this task
# 
# $$ \kappa = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}} $$
# 
# where $w_{i,j}$ is 
# $$ w_{i,j} = \frac{(i-j)^2}{(N-1)^2} $$
# 
# $O$ is N-by-N matrix which is nothing but confusion matrix.
# Actual_value_counts and predicted_value_counts are 1-by-N dimensional vector which is histogram of actula class and predicted class . $E$ is N-by-N matrix which is outer product of Actual_value_counts,predicted_value_counts.
# 
# https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps

# # Load required libreries and methods 

# In[24]:


import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# # Read Data files

# In[25]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_labels = pd.read_csv('train_labels.csv')
specs = pd.read_csv('specs.csv')


# In[3]:


print('training data shape',train.shape)
print('test data shape',test.shape)

unique_id_train = len(train['installation_id'].unique())
unique_id_test = len(test['installation_id'].unique())

print('number of unique installation_id in training data' ,unique_id_train)
print('number of unique installation_id in test data' ,unique_id_test)


# In[26]:


print('training data shape',train.shape)
train.head(3)


# In[5]:


test.head()


# # Data Cleaning
# ### training data cleaning

# In[6]:


# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
# Some of the installation_id from training data do not attempt for Assessment even for a single time.
# First we will get rid of those ids as we will not able to predict the class.
required_id = train[train['type'] == "Assessment"]['installation_id'].drop_duplicates()
train = pd.merge(train, required_id , on="installation_id", how="inner")

unique_id_train = len(train['installation_id'].unique())

print('training data shape',train.shape)
print('number of unique installation_id in training data',unique_id_train)


# * Number of training data reduced to 8.2 M from 11.3 M and number of installation_id reduced to 4242 from 17000.

# ### test data cleaning

# In[7]:


# In test data also we have some installation id that attempt for Assessment but corresponding event_code is not 4100 or 4110(Bird Measurer)
test[(test['installation_id']=='017c5718') & (test['type']=='Assessment')]


# Though event type is Assessment but corresponding event_code is not 4100 or 4110 . So we won't able to predict the class for this installation_id.
# we can simply remove those id or we need to make an assumption. I assumed that those ids belongs to class 0

# # Exploratory Data Analysis
# ### Data preparation

# In[8]:


# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
# In the reference blog they have used two separate function for featurization. I slightly modified and compressed it in only one function.
def get_features(installation_id , dataframe_for_an_id , test_flag=False):
    '''
    
    This function will calculate features for train and test data. 
    It will create 4 columns for four unique world(including None) and
    will create 44 columns for 44 unique title and
    will create 4 columns for four unique type and
    will create 42 columns for 42 unique event_code and
    will create 6 more columns for 'total_duration','total_action','correct_count','incorrect_count','accuracy','accuracy_group'
               ---
        total  100 columns 
    
    except total_duration, accuracy and accuracy_group all other features is number of counts of those feature in a game_session
    if test_flag is True then return last entry of list 
    '''
    # temp_dict initialized with keys (100 columns) and value = 0
    features = []
    features.extend(list(set(train['world'].unique()).union(set(test['world'].unique())))) # all unique worlds in train and test data
    features.extend(list(set(train['title'].unique()).union(set(test['title'].unique())))) # all unique title in train and test data
    features.extend(list(set(train['type'].unique()).union(set(test['type'].unique())))) # all unique type in train and test data
    features.extend(list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))) # all unique event_code in train and test data 
    features.extend(['total_duration','total_action','correct_count','incorrect_count','accuracy','accuracy_group'])
    temp_dict = dict.fromkeys(features,0)
    list_of_features = []
    
    
    def get_all_attempt(sample_df):
        '''
        This fuction will return the dataframe which is used to calculate accuracy_group
        '''
        if sample_df['title'].iloc[0] != 'Bird Measurer (Assessment)':
            all_attempt = sample_df.query('event_code == 4100')
        elif sample_df['title'].iloc[0] == 'Bird Measurer (Assessment)':
            all_attempt = sample_df.query('event_code == 4110')
        return all_attempt
    
    for i, sample_df in dataframe_for_an_id.groupby(by = ['game_session']):
        # sample_df is groupby object 
        # In sample_df 'type','title' and 'world' will not change so first entry of those column is piced
        temp_dict['installation_id'] = installation_id
        temp_type = sample_df['type'].iloc[0]
        temp_title = sample_df['title'].iloc[0]
        temp_world = sample_df['world'].iloc[0]
        temp_event_code = Counter(sample_df['event_code'].values)

        session_size = len(sample_df)

        temp_dict[temp_type]+=session_size
        temp_dict[temp_title]+=session_size    # corresponding type , title and world is incremented by session size
        temp_dict[temp_world]+=session_size                 

        for code, code_count in temp_event_code.items():    # corresponding event_code is incremented
            temp_dict[code]+= code_count                  
            
        duration_in_sec = float(sample_df['game_time'].iloc[-1])/1000   # total_duration is duration of game_session in seconds
        temp_dict['total_duration'] += duration_in_sec
        
        action_count_in_game_session = session_size     # total number of action performed in game_session
        temp_dict['total_action'] += action_count_in_game_session  
        
        isAssessment = temp_type == 'Assessment'
        isBirdMeasureAssessment = isAssessment and temp_title == 'Bird Measurer (Assessment)'
        isAssessment_with_code4110 = isBirdMeasureAssessment and 4110 in list(sample_df['event_code'])
        isNonBirdMeasureAssessment = isAssessment and temp_title != 'Bird Measurer (Assessment)'
        isAssessment_with_code4100 = isNonBirdMeasureAssessment and 4100 in list(sample_df['event_code'])
        
        criterion_to_accuracy_group = isAssessment_with_code4110 or isAssessment_with_code4100 
        
        
        if test_flag and isAssessment and (criterion_to_accuracy_group == False):
            temp_dict['accuracy'] = 0           # there are lots of installation_id in test data that attempt for
            temp_dict['accuracy_group'] = 0     # Assessment but not with event_code 4100 or 4110
            list_of_features.append(temp_dict)  # So I assumed those id belongs to class 0
            
            
        if criterion_to_accuracy_group == False:
            continue
        
        # below section is only performed when criterion_to_accuracy_group is True
        
        all_attempt = get_all_attempt(sample_df)
        correct_count = all_attempt['event_data'].str.contains('true').sum()     
        incorrect_count = all_attempt['event_data'].str.contains('false').sum()
        temp_dict['correct_count'] = correct_count  
        temp_dict['incorrect_count'] = incorrect_count

        if correct_count == 0 and incorrect_count == 0:
            temp_dict['accuracy'] = 0
        else:
            temp_dict['accuracy'] = correct_count/(correct_count + incorrect_count)

        if temp_dict['accuracy']==1:
            temp_dict['accuracy_group']=3
        elif temp_dict['accuracy']==0.5:
            temp_dict['accuracy_group']=2
        elif temp_dict['accuracy']==0:
            temp_dict['accuracy_group']=0
        else :
            temp_dict['accuracy_group']=1

        list_of_features.append(temp_dict)
        temp_dict = dict.fromkeys(features,0)
        
        
    if test_flag:                    # If given data is from test data then return only the last entry of the list
        return list_of_features[-1]
    
    return list_of_features
    


# In[9]:


# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
# below is testing code to check whether get_features function works properly for training data
sample_df = train[train.installation_id == "0006a69f"]
list_of_feature = get_features(sample_df)
list_of_feature
temp_df = pd.DataFrame(list_of_feature)
temp_df


# In[9]:


# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
# below is testing code to check whether get_features function works properly for test data
sample_df = train[train.installation_id == "0006a69f"]
list_of_feature = get_features('0006a69f',sample_df,True)
list_of_feature
temp_df = pd.DataFrame(list_of_feature,index=[0])
temp_df


# * All good , get_features function is working

# ### Get data for EDA
# #### Prepare training data

# In[26]:


# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
final_training_data_list = []
training_groupby_id = train.groupby(by=['installation_id']) 

for installation_id , df_with_unique_id in tqdm(training_groupby_id):
    final_training_data_list.extend(get_features(df_with_unique_id))

final_training_data = pd.DataFrame(final_training_data_list)

#final_training_data.to_csv('final_training_data.csv', index=False)


# In[27]:


print('final_training_data shape :',final_training_data.shape)
final_training_data.head()


# #### Prepare test data

# In[11]:


# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
final_test_data_list = []
test_groupby_id = test.groupby(by=['installation_id'])
for installation_id , df_with_unique_id in tqdm(test_groupby_id):
    final_test_data_list.append(get_features(installation_id,df_with_unique_id,True))
final_test_data = pd.DataFrame(final_test_data_list)


# In[3]:


#final_test_data.to_csv('final_test_data.csv', index=False)
final_test_data = pd.read_csv('final_test_data.csv')
print('final_test_data shape :',final_test_data.shape)
final_test_data.head()


# ### List of content for EDA
# * Bar plot for Count of entries correstponds to each class in training data
# * Bar plot for Count of entries correstponds to each class in test data
# * Stacked bar plot for feature type for each class in training data
# * Stacked bar plot for feature type for each class in test data
# * Stacked bar plot for feature world for each class in training data
# * Stacked bar plot for feature world for each class in test data
# * Barplot for counts for all titles for each class in training data
# * Barplot for counts for all event_code for each class in training data
# * Analysis of feature total duration
# * Analysis of feature total action
# * Analysis of feature incorrect count
# * Tsne plot
# 

# In[13]:


# Data frame for each label for training data
groups_by_label = final_training_data.groupby(by='accuracy_group')
training_data_with_label_0 = groups_by_label.get_group(0) 
training_data_with_label_1 = groups_by_label.get_group(1)
training_data_with_label_2 = groups_by_label.get_group(2)
training_data_with_label_3 = groups_by_label.get_group(3)

# Data frame for each label for test data
groups_by_label_test = final_test_data.groupby(by='accuracy_group')
test_data_with_label_0 = groups_by_label_test.get_group(0) 
test_data_with_label_1 = groups_by_label_test.get_group(1)
test_data_with_label_2 = groups_by_label_test.get_group(2)
test_data_with_label_3 = groups_by_label_test.get_group(3)


# ### Bar plot for Count of entries correstponds to each class in training data

# In[228]:


# reference : https://stackoverflow.com/questions/52080991/display-percentage-above-bar-chart-in-matplotlib
sns.set(style="darkgrid")
plt.figure(figsize = (7,7))
ax = sns.countplot(x = "accuracy_group", data = final_training_data, orient = 'V' )
plt.title('Count of entries in each label in training data')
plt.xlabel('labels')

total = len(final_training_data)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2 - 0.1
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x,y))
plt.show()


# ### Observation
# training data is imbalanced with respect to labels as we can see half of the training data belongs to class 3 and other half is from class 0, 1 and 2.

# ### Bar plot for Count of entries correstponds to each class in test data

# In[229]:


# reference : https://stackoverflow.com/questions/52080991/display-percentage-above-bar-chart-in-matplotlib
sns.set(style="darkgrid")
plt.figure(figsize = (7,7))
ax = sns.countplot(x = "accuracy_group", data = final_test_data, orient = 'V' )
plt.title('Count of entries in each label in test data')
plt.xlabel('labels')

total = len(final_test_data)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()/2 - 0.1
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x,y))
plt.show()


# ### Observation
# Lagre Number of data belongs to class 0 (around 70%) and only 3.9% data for class 1 and 4.1% data for class 2. 

# ### Stacked bar plot for feature 'type' for each class in training data
# 

# In[89]:


# reference : https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
sns.set(style="darkgrid")
plt.figure(figsize = (7,7))
labels = [0,1,2,3]
event_types = ['Game','Activity','Clip','Assessment']
Game = []
Activity = []
Clip = []
Assessment = []
colors = ['crimson','cadetblue','black','darkgray']


for label in labels:
    Game.append(groups_by_label.get_group(label)['Game'].sum())
    Activity.append(groups_by_label.get_group(label)['Activity'].sum())
    Clip.append(groups_by_label.get_group(label)['Clip'].sum())
    Assessment.append(groups_by_label.get_group(label)['Assessment'].sum())

bottom2 = np.array(Game) + np.array(Activity)
bottom3 = bottom2 + np.array(Clip)

Gamebar       = plt.bar(x=labels, height=Game       ,width = 0.5, color = colors[0],label = 'Game')
Activitybar   = plt.bar(x=labels, height=Activity   ,width = 0.5, color = colors[1],label = 'Activity'  ,bottom = Game)
Clipbar       = plt.bar(x=labels, height=Clip       ,width = 0.5, color = colors[2],label = 'Clip'      ,bottom = bottom2)
Assessmentbar = plt.bar(x=labels, height=Assessment ,width = 0.5, color = colors[3],label = 'Assessment',bottom = bottom3)

plt.title('Distribution of feature type in Training data')
plt.xlabel('labels')
plt.ylabel('counts')
plt.xticks(labels)
plt.legend()
plt.show()


# ## Observation:
# * Game type Clip is almost negligible in all labels
# * Most of the event is performed for label 3

# ### Stacked bar plot for feature 'type' for each class in test data

# In[88]:


# reference : https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
sns.set(style="darkgrid")
plt.figure(figsize = (7,7))
labels = [0,1,2,3]
event_types = ['Game','Activity','Clip','Assessment']
Game = []
Activity = []
Clip = []
Assessment = []
colors = ['crimson','cadetblue','black','darkgray']


for label in labels:
    Game.append(groups_by_label_test.get_group(label)['Game'].sum())
    Activity.append(groups_by_label_test.get_group(label)['Activity'].sum())
    Clip.append(groups_by_label_test.get_group(label)['Clip'].sum())
    Assessment.append(groups_by_label_test.get_group(label)['Assessment'].sum())
    
bottom2 = np.array(Game) + np.array(Activity)
bottom3 = bottom2 + np.array(Clip)

Gamebar       = plt.bar(x=labels, height=Game       ,width = 0.5, color = colors[0],label = 'Game')
Activitybar   = plt.bar(x=labels, height=Activity   ,width = 0.5, color = colors[1],label = 'Activity'  ,bottom = Game)
Clipbar       = plt.bar(x=labels, height=Clip       ,width = 0.5, color = colors[2],label = 'Clip'      ,bottom = bottom2)
Assessmentbar = plt.bar(x=labels, height=Assessment ,width = 0.5, color = colors[3],label = 'Assessment',bottom = bottom3)

plt.title('Distribution of feature type in test data')
plt.xlabel('labels')
plt.ylabel('counts')
plt.xticks(labels)
plt.legend()
plt.show()


# ### Observation
# * Game type Clip is almost negligible for each label in test data.
# * Large number of Activity event is performed for label 0 data.

# ### Stacked bar plot for feature 'world' for each class in training data

# In[86]:


# reference : https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
sns.set(style="darkgrid")
plt.figure(figsize = (7,7))
labels = [0,1,2,3]
CRYSTALCAVES = []
TREETOPCITY = []
NONE = []
MAGMAPEAK = []
colors = ['darkblue','darkcyan','darkgoldenrod','darkgray']


for label in labels:
    CRYSTALCAVES.append(groups_by_label.get_group(label)['CRYSTALCAVES'].sum())
    TREETOPCITY.append(groups_by_label.get_group(label)['TREETOPCITY'].sum())
    NONE.append(groups_by_label.get_group(label)['NONE'].sum())
    MAGMAPEAK.append(groups_by_label.get_group(label)['MAGMAPEAK'].sum())

bottom2 = np.array(TREETOPCITY) + np.array(CRYSTALCAVES)
bottom3 = bottom2 + np.array(NONE)
 

CRYSTALCAVESbar = plt.bar(x=labels, height=CRYSTALCAVES, width = 0.5, color = colors[0],label = 'CRYSTALCAVES')
TREETOPCITYbar  = plt.bar(x=labels, height=TREETOPCITY,  width = 0.5, color = colors[1],label = 'TREETOPCITY',bottom = CRYSTALCAVES,)
NONEbar         = plt.bar(x=labels, height=NONE,         width = 0.5, color = colors[2],label = 'NONE'       ,bottom = bottom2)
MAGMAPEAKbar    = plt.bar(x=labels, height=MAGMAPEAK,    width = 0.5, color = colors[3],label = 'MAGMAPEAK'  ,bottom = bottom3)

plt.title('Distribution of feature world in Training data')
plt.xlabel('labels')
plt.ylabel('counts')
plt.xticks(labels)
plt.legend()
plt.show()


# # Observation
# * The world category NONE is negligible for all the labels.
# * Other than NONE, rest of the worlds are almost equaly(one third) distributed for each label.

# In[139]:


# reference : https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
sns.set(style="darkgrid")
plt.figure(figsize = (7,7))
labels = [0,1,2,3]
CRYSTALCAVES = []
TREETOPCITY = []
NONE = []
MAGMAPEAK = []
colors = ['darkblue','darkcyan','darkgoldenrod','darkgray']


for label in labels:
    CRYSTALCAVES.append(groups_by_label_test.get_group(label)['CRYSTALCAVES'].sum())
    TREETOPCITY.append(groups_by_label_test.get_group(label)['TREETOPCITY'].sum())
    NONE.append(groups_by_label_test.get_group(label)['NONE'].sum())
    MAGMAPEAK.append(groups_by_label_test.get_group(label)['MAGMAPEAK'].sum())

bottom2 = np.array(TREETOPCITY) + np.array(CRYSTALCAVES)
bottom3 = bottom2 + np.array(NONE)
 

CRYSTALCAVESbar = plt.bar(x=labels, height=CRYSTALCAVES, width = 0.5, color = colors[0],label = 'CRYSTALCAVES')
TREETOPCITYbar  = plt.bar(x=labels, height=TREETOPCITY,  width = 0.5, color = colors[1],label = 'TREETOPCITY',bottom = CRYSTALCAVES)
NONEbar         = plt.bar(x=labels, height=NONE,         width = 0.5, color = colors[2],label = 'NONE'       ,bottom = bottom2)
MAGMAPEAKbar    = plt.bar(x=labels, height=MAGMAPEAK,    width = 0.5, color = colors[3],label = 'MAGMAPEAK'  ,bottom = bottom3)

plt.title('Distribution of feature world in test data')
plt.xlabel('labels')
plt.ylabel('counts')
plt.xticks(labels)
plt.legend()
plt.show()


# # Observation :
# * In test data also the world NONE is negligible for all the label.
# * As training data here also worlds are distributed equaly (almost one third) in all the labels though number of data in label 1 and label 2 is very less.  

# ### Barplot for counts for all titles for each class in training data

# In[181]:


# prepare a dataframe title_df which contaions summation of event_title values for each label
temp_list = []
labels = [0,1,2,3]
title_list = list(set(train['title'].unique()).union(set(test['title'].unique())))

for label in labels:
    for title in title_list:
        temp_list.append(groups_by_label.get_group(label)[title].sum())
        
title_data = np.array(temp_list).reshape(4,-1)
title_df = pd.DataFrame(data = title_data, columns=title_list)


# In[614]:


# reference : https://stackoverflow.com/questions/22635110/sorting-the-order-of-bars-in-pandas-matplotlib-bar-plots
# prepare ordered data frame
total = title_df.sum(axis=0)
sort_index = np.argsort(total.values)
ordered_title_list = title_df.columns[sort_index]
ordered_title_df = title_df[ordered_title_list]
ordered_title_df.head()


# In[214]:


# reference:https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
# reference : https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.barh.html
sns.set(style="darkgrid")
plt.figure(figsize = (12,12))
labels = [0,1,2,3]
title_label0 = ordered_title_df.iloc[0].values
title_label1 = ordered_title_df.iloc[1].values
title_label2 = ordered_title_df.iloc[2].values
title_label3 = ordered_title_df.iloc[3].values
colors = ['darkblue','darkcyan','darkgoldenrod','darkgray']

bottom2 = np.array(title_label0) + np.array(title_label1)
bottom3 = bottom2 + np.array(title_label2)
 
title_label0bar = plt.barh(y=ordered_title_list, width=title_label0, height = 0.5, color = colors[0],label = 'label_0')
title_label1bar = plt.barh(y=ordered_title_list, width=title_label1, height = 0.5, color = colors[1],label = 'label_1', left = title_label0,)
title_label2bar = plt.barh(y=ordered_title_list, width=title_label2, height = 0.5, color = colors[2],label = 'label_2', left = bottom2)
title_label3bar = plt.barh(y=ordered_title_list, width=title_label3, height = 0.5, color = colors[3],label = 'label_3', left = bottom3)

plt.title('Distribution of feature title in train data')
plt.ylabel('event_title')
plt.xlabel('counts')
plt.yticks(ordered_title_list)
plt.legend()
plt.show()
        


# * Let's plot another stacked plot for least occure event title

# In[621]:


# reference:https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
# reference : https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.barh.html
least_occure_title_list = ordered_title_df.columns[0:20]
least_occure_title_df = ordered_title_df[least_occure_title_list]

sns.set(style="darkgrid")
plt.figure(figsize = (12,12))
labels = [0,1,2,3]
title_label0 = least_occure_title_df.iloc[0].values
title_label1 = least_occure_title_df.iloc[1].values
title_label2 = least_occure_title_df.iloc[2].values
title_label3 = least_occure_title_df.iloc[3].values
colors = ['darkblue','darkcyan','darkgoldenrod','darkgray']

bottom2 = np.array(title_label0) + np.array(title_label1)
bottom3 = bottom2 + np.array(title_label2)
 
title_label0bar = plt.barh(y=least_occure_title_list, width=title_label0, height = 0.5, color = colors[0],label = 'label_0')
title_label1bar = plt.barh(y=least_occure_title_list, width=title_label1, height = 0.5, color = colors[1],label = 'label_1', left = title_label0,)
title_label2bar = plt.barh(y=least_occure_title_list, width=title_label2, height = 0.5, color = colors[2],label = 'label_2', left = bottom2)
title_label3bar = plt.barh(y=least_occure_title_list, width=title_label3, height = 0.5, color = colors[3],label = 'label_3', left = bottom3)

plt.title('Distribution of feature title in train data')
plt.ylabel('event_title')
plt.xlabel('counts')
plt.yticks(least_occure_title_list)
plt.legend()
plt.show()


# ### Observation
# * Almost half of the event title are negligible as it occures very few times.
# * most of the event title is Activity from top 24 event titles.
# * Label 3 and Label 0 Contains more number of actions as compared to label 1 and label 2.
# * bottle filter activity is the most pop-up event and the Treasure Map is least pop-up event title which is 0.3% of most pop-up event 

# ### Barplot for counts for all event_code for each class in training data

# In[14]:


# prepare a dataframe title_df which contaions summation of event_code values for each label
temp_list = []
labels = [0,1,2,3]
event_code_list = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

for label in labels:
    for event_code in event_code_list:
        temp_list.append(groups_by_label.get_group(label)[str(event_code)].sum())
        
event_code_data = np.array(temp_list).reshape(4,-1)
event_code_df = pd.DataFrame(data = event_code_data, columns=event_code_list)


# In[15]:


# reference : https://stackoverflow.com/questions/22635110/sorting-the-order-of-bars-in-pandas-matplotlib-bar-plots
# prepare ordered data frame
total = event_code_df.sum(axis=0)
sort_index = np.argsort(total.values)
ordered_event_code_list = event_code_df.columns[sort_index]
ordered_event_code_df = event_code_df[ordered_event_code_list]

code_list = []
for i in ordered_event_code_list:
    code_list.append(str(i))
    
ordered_event_code_df.head()


# In[16]:


# reference:https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
# reference: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.barh.html
sns.set(style="darkgrid")
plt.figure(figsize = (12,12))
labels = [0,1,2,3]
event_code_label0 = ordered_event_code_df.iloc[0].values
event_code_label1 = ordered_event_code_df.iloc[1].values
event_code_label2 = ordered_event_code_df.iloc[2].values
event_code_label3 = ordered_event_code_df.iloc[3].values
colors = ['darkblue','darkcyan','darkgoldenrod','darkgray']

bottom2 = np.array(event_code_label0) + np.array(event_code_label1)
bottom3 = bottom2 + np.array(event_code_label2)
 
event_code_label0bar = plt.barh(y=code_list, width=event_code_label0, height = 0.5, color = colors[0],label = 'label_0')
event_code_label1bar = plt.barh(y=code_list, width=event_code_label1, height = 0.5, color = colors[1],label = 'label_1', left = event_code_label0)
event_code_label2bar = plt.barh(y=code_list, width=event_code_label2, height = 0.5, color = colors[2],label = 'label_2', left = bottom2)
event_code_label3bar = plt.barh(y=code_list, width=event_code_label3, height = 0.5, color = colors[3],label = 'label_3', left = bottom3)

plt.title('Distribution of feature event_code in train data')
plt.ylabel('event_code')
plt.xlabel('counts')
plt.yticks(code_list)
plt.legend()
plt.show()


# * Let's plot another stacked plot for least occure event_code

# In[23]:


# reference:https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
# reference : https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.barh.html
least_occure_event_code_list = ordered_event_code_df.columns[0:23]
least_occure_event_code_df = ordered_event_code_df[least_occure_event_code_list]

code_list = []
for i in least_occure_event_code_list:
    code_list.append(str(i))

sns.set(style="darkgrid")
plt.figure(figsize = (12,12))
labels = [0,1,2,3]
event_code_label0 = least_occure_event_code_df.iloc[0].values
event_code_label1 = least_occure_event_code_df.iloc[1].values
event_code_label2 = least_occure_event_code_df.iloc[2].values
event_code_label3 = least_occure_event_code_df.iloc[3].values
colors = ['darkblue','darkcyan','darkgoldenrod','darkgray']

bottom2 = np.array(event_code_label0) + np.array(event_code_label1)
bottom3 = bottom2 + np.array(event_code_label2)
 
event_code_label0bar = plt.barh(y=code_list, width=event_code_label0, height = 0.5, color = colors[0],label = 'label_0')
event_code_label1bar = plt.barh(y=code_list, width=event_code_label1, height = 0.5, color = colors[1],label = 'label_1', left = event_code_label0,)
event_code_label2bar = plt.barh(y=code_list, width=event_code_label2, height = 0.5, color = colors[2],label = 'label_2', left = bottom2)
event_code_label3bar = plt.barh(y=code_list, width=event_code_label3, height = 0.5, color = colors[3],label = 'label_3', left = bottom3)

code_list = []
for i in least_occure_event_code_list:
    code_list.append(str(i))

plt.title('Distribution of feature event_code in train data')
plt.ylabel('event_code')
plt.xlabel('counts')
plt.yticks(code_list)
plt.legend()
plt.show()


# ### Observation 
# * Around 60% of total event_code holds top five event_code.
# * Half of the event_code is below 5% of total counts.
# * 4970 is the most pop-up event_code and the 4080 is least pop-up event_code which is less than 0.04% of most pop-up event_code.

# In[279]:


# reference : https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
sns.set(style="darkgrid")
plt.figure(figsize = (20,10))
temp_array = np.array(ordered_event_code_df.sum())
sum_total = ordered_event_code_df.sum().sum()
ax = plt.plot()
for i in range(1,43):
    total = 0
    for j in range(i):
        total+=temp_array[j]
    plt.scatter(i,total)
    percent = str(np.round((total/sum_total)*100,1))+'%'
    if i%2==0:
        plt.annotate(percent,(i-0.4,total+10**5.4))  # this if else condition to annotate above and below of the point
    else:
        plt.annotate(percent,(i-0.4,total-10**5.5))
plt.xticks(range(1,43))
plt.title('event_code vs cumulative counts')
plt.xlabel('events')
plt.ylabel('Cumulative sum of events')
plt.show()


# ### Observation
# * Here it is easy to see the event contribution on total counts. We can see that half of the event (21th event)
# contributes only 3.4 % of total counts.
# 

# ### Total duration analysis
# Total duration is the duration in seconds starts from begining of a game_session till the time an user attempts an Assessment event.

# ### total duration analysis for label 0

# In[307]:


# reference : https://www.w3schools.com/python/python_ml_scatterplot.asp
total_duration_label_0 = training_data_with_label_0['total_duration'].values
sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
for i in np.arange(1,101,1):
    plt.scatter(i,np.percentile(total_duration_label_0,i))
plt.title('Percentile vs Percentile value for feature total duration of label 0')
plt.xlabel('percentile')
plt.ylabel('percentile value')
plt.xticks(np.arange(0,110,10))
plt.show()


# * Let's exclude 100th percentile
#     

# In[308]:


# reference : https://www.w3schools.com/python/python_ml_scatterplot.asp
sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
for i in np.arange(1,100,1):
    plt.scatter(i,np.percentile(total_duration_label_0,i))
plt.title('Percentile vs Percentile value for feature total duration of label 0')
plt.xlabel('percentile')
plt.ylabel('percentile value')
plt.xticks(np.arange(0,110,10))
plt.show()


# ### Observation
# * 90% of total_duration are below 2000 sec.

# ### total duration analysis for label 1

# In[309]:


# reference : https://www.w3schools.com/python/python_ml_scatterplot.asp
total_duration_label_1 = training_data_with_label_1['total_duration'].values
sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
for i in np.arange(1,101,1):
    plt.scatter(i,np.percentile(total_duration_label_1,i))
plt.title('Percentile vs Percentile value for feature total duration of label 1')
plt.xlabel('percentile')
plt.ylabel('percentile value')
plt.xticks(np.arange(0,110,10))
plt.show()


# let's exclude 100th percentile

# In[310]:


# reference : https://www.w3schools.com/python/python_ml_scatterplot.asp
sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
for i in np.arange(1,100,1):
    plt.scatter(i,np.percentile(total_duration_label_1,i))
plt.title('Percentile vs Percentile value for feature total duration of label 1')
plt.xlabel('percentile')
plt.ylabel('percentile value')
plt.xticks(np.arange(0,110,10))
plt.show()


# ### Observation 
# Here 90th percentile value is 1300 second which is lower than label 0's 90th percentile value.

# ### total duration analysis for label 2

# In[312]:


# reference : https://www.w3schools.com/python/python_ml_scatterplot.asp
total_duration_label_2 = training_data_with_label_2['total_duration'].values
sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
for i in np.arange(1,100,1):
    plt.scatter(i,np.percentile(total_duration_label_2,i))
plt.title('Percentile vs Percentile value for feature total duration of label 2')
plt.xlabel('percentile')
plt.ylabel('percentile value')
plt.xticks(np.arange(0,110,10))
plt.show()


# ### Observation
# * looks very much similar as label 2 total duration.

# ### total duration analysis for label 3

# In[313]:


# reference :https://www.w3schools.com/python/python_ml_scatterplot.asp
total_duration_label_3 = training_data_with_label_3['total_duration'].values
sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
for i in np.arange(1,100,1):
    plt.scatter(i,np.percentile(total_duration_label_3,i))
plt.title('Percentile vs Percentile value for feature total duration of label 3')
plt.xlabel('percentile')
plt.ylabel('percentile value')
plt.xticks(np.arange(0,110,10))
plt.show()


# ### Conclusion
# * geting percentile value for total duration of each label is not convincing as we are not able to see any mejor difference in there percentile value.
# * Let's see the distribution plot for total duration.

# ### total duration distribution plot for each label

# In[334]:


# reference : https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# reference : https://stackoverflow.com/questions/45911709/limit-the-range-of-x-in-seaborn-distplot-kde-estimation
sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
sns.kdeplot(total_duration_label_3, shade=True,clip=(0.0, 30000.0),label = 'total_duration_label_3',color = 'red')
sns.kdeplot(total_duration_label_2, shade=True,clip=(0.0, 30000.0),label = 'total_duration_label_2',color = 'green')
sns.kdeplot(total_duration_label_1, shade=True,clip=(0.0, 30000.0),label = 'total_duration_label_1',color = 'yellow')
sns.kdeplot(total_duration_label_0, shade=True,clip=(0.0, 30000.0),label = 'total_duration_label_0',color = 'blue')
plt.title('KDE plot for feature total duration for each label')
plt.xlabel('total duration')
plt.ylabel('probability dencity')
plt.show()


# ### Conclusion
# * Though the clip parameter is not preferable in kdeplot, I have used it just to better visualize the kde plot.
# * Probability density is highly overlapped for total duration of each label in training data. total duration might not very helpful for classification task.

# ### Analysis of feature total action

# In[336]:


# reference : https://seaborn.pydata.org/generated/seaborn.kdeplot.html
total_action_label_0 = training_data_with_label_0['total_action'].values
total_action_label_1 = training_data_with_label_1['total_action'].values
total_action_label_2 = training_data_with_label_2['total_action'].values
total_action_label_3 = training_data_with_label_3['total_action'].values

sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
sns.kdeplot(total_action_label_3, shade=True,label = 'total_action_label_3', color = 'red')
sns.kdeplot(total_action_label_2, shade=True,label = 'total_action_label_2', color = 'green')
sns.kdeplot(total_action_label_1, shade=True,label = 'total_action_label_1', color = 'yellow')
sns.kdeplot(total_action_label_0, shade=True,label = 'total_action_label_0', color = 'blue')
plt.title('KDE plot for feature total action for each label')
plt.xlabel('total action')
plt.ylabel('probability dencity')
plt.show()


# ### conclusion
# * Like total duration here also in total action probability density are highly overlapped for all the labels in training data. The feature total action might not very useful for classification.

# ### analysis of feature incorrect_count
# * For label 3 all incorrect count will be 0.
# * For label 2 all incorrect count will be 1.
# * so we will visualize for label 2 and 3. 

# In[337]:


# reference :https://seaborn.pydata.org/generated/seaborn.kdeplot.html
incorrect_count_label_0 = training_data_with_label_0['incorrect_count'].values
incorrect_count_label_1 = training_data_with_label_1['incorrect_count'].values

sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
sns.kdeplot(incorrect_count_label_0, shade=True,label = 'incorrect_count_label_0', color = 'red')
sns.kdeplot(incorrect_count_label_1, shade=True,label = 'incorrect_count_label_1', color = 'green')

plt.title('KDE plot for feature incorrect_count for each label')
plt.xlabel('incorrect_count')
plt.ylabel('probability dencity')
plt.show()


# ### observation
# * diviation from mean is more for label 0 as comapared to label 1.

# ## TSNE plot
# * I am using first 10,000 data points from training data for tsne plot just to reduce the complexity.

# In[362]:


sample_training_data = final_training_data[0:10000]
train_label = sample_training_data['accuracy_group'].values
sample_training_data = sample_training_data.drop(['accuracy_group'], axis=1)
train_data_matrix = sample_training_data.to_numpy()


# In[363]:


# reference : https://medium.com/@garora039/dimensionality-reduction-using-t-sne-effectively-cabb2cd519b
# tsne with perplexity = 50
tsne = TSNE(n_components = 2, perplexity = 50)
X_embedding = tsne.fit_transform(train_data_matrix)
for_tsne = np.hstack((X_embedding,np.array(train_label).reshape(-1,1)))
tsne_df = pd.DataFrame(data = for_tsne, columns = ['dimension_x','dimension_y','score'])

sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
sns.FacetGrid(tsne_df, hue='score', height = 10).map(plt.scatter,'dimension_x','dimension_y').add_legend()
plt.title('tsne plot with perplexity = 50')
plt.xlabel('reduced_dimension1')
plt.ylabel('reduced_dimension2')
plt.show()


# In[364]:


# reference : https://medium.com/@garora039/dimensionality-reduction-using-t-sne-effectively-cabb2cd519b
# tsne with perplexity = 20
tsne = TSNE(n_components = 2, perplexity = 20)
X_embedding = tsne.fit_transform(train_data_matrix)
for_tsne = np.hstack((X_embedding,np.array(train_label).reshape(-1,1)))
tsne_df = pd.DataFrame(data = for_tsne, columns = ['dimension_x','dimension_y','score'])

sns.set(style="darkgrid")
plt.figure(figsize = (15,10))
sns.FacetGrid(tsne_df, hue='score', height = 10).map(plt.scatter,'dimension_x','dimension_y').add_legend()
plt.title('tsne plot with perplexity = 20')
plt.xlabel('reduced_dimension1')
plt.ylabel('reduced_dimension2')
plt.show()


# ## Conclusion
# * For both perplexity = 20 and perplexity = 50 we got highly overlapped tsne plot. There is no separate cluster for an unique label.
# 
# <u> **It is going to very tough to perform classification task on this data set.** </u> 

# # Data Preparation
# * I have used XBGClassifier a lot and almost every time it performs best as compared to other linear model when data set contains less number of fetures. This is why I am choosing XGBClassifier to start with.
# 
# ### Train test split

# In[6]:


# reference : https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps

def calculate_QWK(actual_label,predicted_label):
    '''
    this function will calculate quadratic weighted kappa given actual 
    and predicted label array.
    '''
    N = 4 # unique labels
    hist_actual_label = np.zeros(N)
    hist_predicted_label = np.zeros(N)
    w = np.zeros((N,N))
    numerator = 0       # w and O
    denominator = 0     # w and E
    
    conf_mat = confusion_matrix(actual_label,predicted_label)

    for i in actual_label:               # this part will calculate histogram for actual and predicted label
        hist_actual_label[i]+=1
    for j in predicted_label:
        hist_predicted_label[j]+=1

    E = np.outer(hist_actual_label, hist_predicted_label)  # E is N-by-N matrix which is outer product of 
                                                           # histogram of actual and predicted label    
    for i in range(N):                   # w is N-by-N matrix which is calculated by the given expression
        for j in range(N):
            w[i][j] = (i-j)**2/((N-1)**2)

    E = E/E.sum()
    O = conf_mat/conf_mat.sum()  # normalize confusion matrix and E

    for i in range(N):
        for j in range(N):                # this section calculates numerator and denominator 
            numerator+=w[i][j]*O[i][j]
            denominator+=w[i][j]*E[i][j]

    kappa = 1-numerator/denominator
    
    return kappa


# In[7]:


# reference : Assignment 19 Malware detection Problem
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    print("Number of misclassified points ",(len(test_y)-np.trace(C))/len(test_y)*100)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j
    
    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column
    
    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1
    
    B =(C/C.sum(axis=0))
    #divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]] 
    
    labels = [0,1,2,3]
    cmap=sns.light_palette("blue")
    # representing A in heatmap format
    print("-"*35, "Confusion matrix", "-"*35)
    plt.figure(figsize=(8,6))
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-"*35, "Precision matrix", "-"*35)
    plt.figure(figsize=(8,6))
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    print("Sum of columns in precision matrix",B.sum(axis=0))
    
    # representing B in heatmap format
    print("-"*35, "Recall matrix"    , "-"*35)
    plt.figure(figsize=(8,6))
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    print("Sum of rows in precision matrix",A.sum(axis=1))


# In[8]:


## testing code for function calculate_QWK()
actual_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])
predicted_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])
print('QWK when actual_label and predicted_label are same is :',calculate_QWK(actual_label_temp,predicted_label_temp))

actual_label_temp = np.array([0,3,0,3,2,0,3,1,2,3,0])
predicted_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])
print('QWK when actual_label and predicted_label are different is :',calculate_QWK(actual_label_temp,predicted_label_temp))


# In[17]:


## if we include features 'correct_count','incorect_count' and 'accuracy' to train a model then
## it will become a trvial task like if-else condition to predict the label that we dont want. 
## we calculated 'correct_count','incorect_count' and 'accuracy' to get the label of training  and test data
## but we want our model to predict the label without those feature thats why we will remove those feature.
X = final_training_data.copy()
X_test = final_test_data.copy()
y = X['accuracy_group'].values
y_test = X_test['accuracy_group'].values


X = X.drop(['correct_count','incorrect_count','accuracy','accuracy_group'], axis=1)
X_test = X_test.drop(['correct_count','incorrect_count','accuracy','accuracy_group'],axis =1)

X_train, X_cv, y_train, y_cv = train_test_split(X, y,stratify=y,test_size=0.2)
X_train = X_train.values
X_cv = X_cv.values
X_test = X_test.values

print('size of training data and labels :',X_train.shape,y_train.shape)
print('size of cv data and labels :',X_cv.shape,y_cv.shape)
print('size of test data and labels :',X_test.shape,y_test.shape)


# # XGBClassifier
# 
# ### hyper parameter tuning

# In[18]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time
from sklearn.metrics import make_scorer


# In[19]:


# reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
start = time.time()
params = {'max_depth':[5,6,7,9,11,13],
          'min_child_weight':[0.001,0.01,0.1,1],
          'n_estimators':[50,100,200,500,600,700]}

QWK_scorer = make_scorer(calculate_QWK, greater_is_better=True)

model  = xgb.XGBClassifier(booster='gbtree')
grid = RandomizedSearchCV(model, param_distributions=params, scoring = QWK_scorer,                     n_jobs=-1,cv=5,return_train_score=True) 
                                                
grid.fit(X_train,y_train) 
print('time taken to train the model in sec:',time.time() - start)


# In[20]:


grid.best_estimator_


# In[21]:


xgb_model = grid.best_estimator_
xgb_model.fit(X_train,y_train)


# In[22]:


actual_label = y_train
predicted_label = xgb_model.predict(X_train)
print('Quadratic weighed kappa for training data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_cv
predicted_label = xgb_model.predict(X_cv)
print('Quadratic weighed kappa for cross validation data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_test
predicted_label = xgb_model.predict(X_test)
print('Quadratic weighed kappa for test data is :',calculate_QWK(actual_label,predicted_label))

plot_confusion_matrix(actual_label,predicted_label)


# In[23]:


pkl_filename = "final_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(xgb_model, file)


# ## Observation 
# * Along with quadratic weighted kappa it is always better to look at the precesion and recall metrics because it gives better understanding of misclassification.
# * In the recall metrix we can see that lots of class1 and class2 data predicted as class3 data poits.
# * Quadratic weighted kappa for test data is 0.864 and percentage of misclassification is 7.7%.

# **note:** 
# * While performing featurization of training data we said that we will calculate correct_count or incorrect_count only for those data having event_type='Assessment' and event_code = either 4100 or 4110(Bird Measurer Assessment) and data point belongs to label 0 only when incorrect_count is = 0.
# * But while performing featurization of test data we have seen that some of the installation_id performs 'Assessment' event but not with the event_code 4100 or 4110 and we assumed that these data will belongs to label 0 with accuracy 0. Otherwise we had to remove those data from test set that we didn't want.
# * The strategy to Decide the label of training and test data is slightly different and this might be the reason for difference in performance metric(QWK) for training (QWK = 1) and test data (QWK = 0.878).

# # Random Forest model
# 
# ### Hyper parameter tuning

# In[32]:


# reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
from sklearn.ensemble import RandomForestClassifier

start = time.time()
params = {'n_estimators':[100,300,400,500,600,700],
          'criterion' : ['gini', 'entropy'],
          'min_samples_split' : [2,3,4,5],
          'max_depth':[5,6,7,10,12,15]}

QWK_scorer = make_scorer(calculate_QWK, greater_is_better=True)

model  = RandomForestClassifier(class_weight='balanced')
grid = RandomizedSearchCV(model, param_distributions=params, scoring = QWK_scorer,                     n_jobs=-1,cv=5,return_train_score=True) 
                                                
grid.fit(X_train,y_train) 
print('time taken to train the model in sec:',time.time() - start)


# In[33]:


grid.best_estimator_


# In[34]:


model = grid.best_estimator_
model.fit(X_train,y_train)


# In[35]:


actual_label = y_train
predicted_label = model.predict(X_train)
print('Quadratic weighed kappa for training data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_cv
predicted_label = model.predict(X_cv)
print('Quadratic weighed kappa for cross validation data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_test
predicted_label = model.predict(X_test)
print('Quadratic weighed kappa for test data is :',calculate_QWK(actual_label,predicted_label))

plot_confusion_matrix(actual_label,predicted_label)


# ## Observation 
# * Along with quadratic weighted kappa it is always better to look at the precesion and recall metrics because it gives better understanding of misclassification.
# * In the recall metrix we can see that lots of class1 and class2 data predicted as class3 data poits.
# * Quadratic weighted kappa for test data is 0.77 and percentage of misclassification is 11.5%.

# # Simple neural network
# 
# ### One Hot Encoding of labels  

# In[539]:


# our last layer in NN will be a softmax layer so let's create one hot encoded vector for labels
#reference https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
y_train_ohe = lb.transform(y_train)
y_test_ohe = lb.transform(y_test)
y_cv_ohe = lb.transform(y_cv)
print(X_train.shape,y_train_ohe.shape)
print(X_test.shape,y_test_ohe.shape)
print(X_cv.shape,y_cv_ohe.shape)


# In[540]:


# scaling the feature before start training the model 
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X_train)
X_train_norm = min_max_scaler.transform(X_train)
X_test_norm = min_max_scaler.transform(X_test)
X_cv_norm = min_max_scaler.transform(X_cv)


# In[541]:


#importing layers from keras
from keras.layers import Dense,concatenate,Activation,Dropout,Input,Flatten
from keras.layers import Embedding
from keras.models import Model
from keras.models import Sequential 
from keras.layers import Dense,Flatten,Embedding,LSTM,Conv1D,Conv2D,BatchNormalization,SpatialDropout1D
from keras.layers import MaxPooling1D
import tensorflow as tf


# ### Model structure

# In[542]:



model = Sequential()

model.add(Dense(units=74,activation='sigmoid',kernel_initializer='he_normal'
                ,input_dim = X_train_norm.shape[1],name="Dense1"))

model.add(Dropout(rate = 0.4))

model.add(Dense(units=22,activation='sigmoid',kernel_initializer="he_normal",name='Dense2'))

model.add(Dropout(rate = 0.6))

#model.add(Dense(units=12,activation='sigmoid',kernel_initializer="he_normal",name='Dense3'))

#model.add(Dropout(rate = 0.3))

model.add(Dense(units=4,activation='softmax',kernel_initializer="glorot_uniform",name='output'))


# In[543]:


model.summary()


# In[544]:


#reference https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
from keras.utils.vis_utils import plot_model
import graphviz
import pydot
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[546]:


from tensorflow.keras.losses import CategoricalCrossentropy
model.compile(optimizer='adam', loss= CategoricalCrossentropy(), metrics=['accuracy'])
class_weights = {0:1.2, 1:2.5 , 2:2.5 , 3:1.2}
score = model.fit(X_train_norm,y_train_ohe
                  ,batch_size= 128
                  ,epochs=30
                  ,class_weight = class_weights
                  ,validation_data=(X_test_norm,y_test_ohe))


# In[547]:


actual_label = y_train
predicted_label = model.predict_classes(X_train_norm)
print('Quadratic weighed kappa for training data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_cv
predicted_label = model.predict_classes(X_cv_norm)
print('Quadratic weighed kappa for cross validation data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_test
predicted_label = model.predict_classes(X_test_norm)
print('Quadratic weighed kappa for test data is :',calculate_QWK(actual_label,predicted_label))

plot_confusion_matrix(actual_label,predicted_label)


# ## Conclusion 
# * Using neural network we got QWK for test data = 0.34 which is not that good.
# * precision for class 2 is 0 which is not desirable.

# # Logistic Regression
# ### steps to perform in this section
# * built a simple Logistic Regression model with default value and get top three important feature based on coefficient value corresponds to the feature.
# * create new feature using box-cox , sin , cos ,binning and product of those top three feature.
# * log transformation of feature 'total_duration' and 'total_action'.
# * Now train a Logistic regression model with hyper parameter tuning.

# ### get important features

# In[376]:


from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression()
LR_model.fit(X_train,y_train)

columns = final_training_data.columns

top_ten_imp_feature_index0 = np.argsort(abs(LR_model.coef_[0]))[-10:]
top_ten_imp_feature0 = columns[top_ten_imp_feature_index0]

top_ten_imp_feature_index1 = np.argsort(abs(LR_model.coef_[1]))[-10:]
top_ten_imp_feature1 = columns[top_ten_imp_feature_index1]

top_ten_imp_feature_index2 = np.argsort(abs(LR_model.coef_[2]))[-10:]
top_ten_imp_feature2 = columns[top_ten_imp_feature_index2]

top_ten_imp_feature_index3 = np.argsort(abs(LR_model.coef_[3]))[-10:]
top_ten_imp_feature3 = columns[top_ten_imp_feature_index3]

imp_features = set(top_ten_imp_feature0) & set(top_ten_imp_feature1) & set(top_ten_imp_feature2) & set(top_ten_imp_feature3)
print('important features are :',imp_features)


# ### get derived feature using box-cox transformation 

# In[402]:


# reference : https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b

from scipy.stats import boxcox
# make a copy of dataframe  
final_training_data_derived = final_training_data.copy()
final_test_data_derived = final_test_data.copy()

# create derived feature for feature['2010']

a,opt_lambda = boxcox(final_training_data['2010'].values+0.001,lmbda=None)# get optimal lambda
box_cox_2010_train = boxcox(final_training_data['2010'].values+0.001,lmbda=opt_lambda) # get derived box_cox feature for training data
final_training_data_derived['box_cox_2010'] = box_cox_2010_train
box_cox_2010_test = boxcox(final_test_data['2010'].values+0.001,lmbda=opt_lambda)# get derived box_cox feature for test data
final_test_data_derived['box_cox_2010'] = box_cox_2010_test

# create derived feature for feature['2020']

a,opt_lambda = boxcox(final_training_data['2020'].values+0.001,lmbda=None)# get optimal lambda
box_cox_2020_train = boxcox(final_training_data['2020'].values+0.001,lmbda=opt_lambda)# get derived box_cox feature for training data
final_training_data_derived['box_cox_2020'] = box_cox_2020_train
box_cox_2020_test = boxcox(final_test_data['2020'].values+0.001,lmbda=opt_lambda)# get derived box_cox feature for test data
final_test_data_derived['box_cox_2020'] = box_cox_2020_test

# create derived feature for feature['2030']

a,opt_lambda = boxcox(final_training_data['2030'].values+0.001,lmbda=None)# get optimal lambda
box_cox_2030_train = boxcox(final_training_data['2030'].values+0.001,lmbda=opt_lambda)# get derived box_cox feature for training data
final_training_data_derived['box_cox_2030'] = box_cox_2030_train
box_cox_2030_test = boxcox(final_test_data['2030'].values+0.001,lmbda=opt_lambda)# get derived box_cox feature for test data
final_test_data_derived['box_cox_2030'] = box_cox_2030_test


# ### get derived feature using sine and cosine function                                

# In[418]:


# reference : https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
# get sin features
final_training_data_derived['sin_2010'] = np.sin(final_training_data['2010'].values)
final_test_data_derived['sin_2010'] = np.sin(final_test_data['2010'].values)

final_training_data_derived['sin_2020'] = np.sin(final_training_data['2020'].values)
final_test_data_derived['sin_2020'] = np.sin(final_test_data['2020'].values)

final_training_data_derived['sin_2030'] = np.sin(final_training_data['2030'].values)
final_test_data_derived['sin_2030'] = np.sin(final_test_data['2030'].values)

# get cosine features
final_training_data_derived['cos_2010'] = np.cos(final_training_data['2010'].values)
final_test_data_derived['cos_2010'] = np.cos(final_test_data['2010'].values)

final_training_data_derived['cos_2020'] = np.cos(final_training_data['2020'].values)
final_test_data_derived['cos_2020'] = np.cos(final_test_data['2020'].values)

final_training_data_derived['cos_2030'] = np.cos(final_training_data['2030'].values)
final_test_data_derived['cos_2030'] = np.cos(final_test_data['2030'].values)


# ### get derived feature using log transformation of feature 'total_duration' and 'total_action'
# * In EDA section we ahve seen that the distribution of feature 'total_duration' and 'total_action' looks some what like lognormal distribution . So here we will add log transformation of those feature.

# In[446]:


# reference : https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
# 0.5 is added to avoid log(0)
final_training_data_derived['log_total_duration'] = np.log(final_training_data['total_duration'].values+0.5)
final_test_data_derived['log_total_duration'] = np.log(final_test_data['total_duration'].values+0.5)

final_training_data_derived['log_total_action'] = np.log(final_training_data['total_action'].values+0.5)
final_test_data_derived['log_total_action'] = np.log(final_test_data['total_action'].values+0.5)
# to see the distribution after transformation
sns.kdeplot(final_test_data_derived['log_total_duration'])
plt.show()


# * we expected a normal distribution but anyway what we got is better that skewed distribution.

# ### get derived feature using Fixed width binning for feature '2010','2020' and '2030'

# In[459]:


print('2010 max = {}, 2010 min = {}'.format(max(final_training_data['2010'].values),min(final_training_data['2010'].values)))


# * we will perform binning for feature '2020' and '2030' only as we can see range is only 0-2 for feature '2010'

# In[462]:


# reference : https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
final_training_data_derived['2020_bins'] = np.array(np.floor(final_training_data['2020'].values) / 20.)
final_test_data_derived['2020_bins'] = np.array(np.floor(final_test_data['2020'].values) / 20.)

final_training_data_derived['2030_bins'] = np.array(np.floor(final_training_data['2030'].values) / 20.)
final_test_data_derived['2030_bins'] = np.array(np.floor(final_test_data['2030'].values) / 20.)


# ### get drived feature using product of feature '2010','2020' and '2030'

# In[468]:


# reference : https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
final_training_data_derived['2010dot2020'] = np.multiply(final_training_data['2010'],final_training_data['2020'])
final_test_data_derived['2010dot2020'] = np.multiply(final_test_data['2010'],final_test_data['2020'])

final_training_data_derived['2010dot2030'] = np.multiply(final_training_data['2010'],final_training_data['2030'])
final_test_data_derived['2010dot2030'] = np.multiply(final_test_data['2010'],final_test_data['2030'])

final_training_data_derived['2020dot2030'] = np.multiply(final_training_data['2020'],final_training_data['2030'])
final_test_data_derived['2020dot2030'] = np.multiply(final_test_data['2020'],final_test_data['2030'])


# In[469]:


print(final_training_data_derived.shape , final_training_data.shape)
print(final_test_data_derived.shape , final_test_data.shape)


# * We have derived 19 new feature from the existing feature

# ### train test split

# In[470]:


## if we include features 'correct_count','incorect_count' and 'accuracy' to train a model then
## it will become a trvial task like if-else condition to predict the label that we dont want. 
## we calculated 'correct_count','incorect_count' and 'accuracy' to get the label of training  and test data but
## we want our model to predict the label without those feature thats why we will remove those feature.
X = final_training_data_derived.copy()
X_test = final_test_data_derived.copy()
y = X['accuracy_group'].values
y_test = X_test['accuracy_group'].values


X = X.drop(['correct_count','incorrect_count','accuracy','accuracy_group'], axis=1)
X_test = X_test.drop(['correct_count','incorrect_count','accuracy','accuracy_group'],axis =1)

X_train, X_cv, y_train, y_cv = train_test_split(X, y,stratify=y,test_size=0.2)
X_train = X_train.values
X_cv = X_cv.values
X_test = X_test.values

print('size of training data and labels :',X_train.shape,y_train.shape)
print('size of cv data and labels :',X_cv.shape,y_cv.shape)
print('size of test data and labels :',X_test.shape,y_test.shape)


# ### min-max scaling

# In[477]:


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X_train)
X_train_norm = min_max_scaler.transform(X_train)
X_test_norm = min_max_scaler.transform(X_test)
X_cv_norm = min_max_scaler.transform(X_cv)


# ### hyper parameter tuning logistic regression model

# In[486]:


# reference : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# reference : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
from sklearn.linear_model import LogisticRegression

start = time.time()
params = {'penalty':['l1','l2'],
          'C' : [0.0001,0.001,0.01,0.1,1,10]}

QWK_scorer = make_scorer(calculate_QWK, greater_is_better=True)

model  = LogisticRegression(multi_class='auto',class_weight='balanced')
grid = RandomizedSearchCV(model, param_distributions=params, scoring = QWK_scorer,                     n_jobs=-1,cv=5,return_train_score=True) 
                                                
grid.fit(X_train_norm,y_train) 
print('time taken to train the model in sec:',time.time() - start)


# In[487]:


grid.best_estimator_


# In[488]:


from sklearn.linear_model import LogisticRegression

start = time.time()
params = {'penalty':['l1','l2'],
          'C' : [0.0001,0.001,0.01,0.1,1,10]}

QWK_scorer = make_scorer(calculate_QWK, greater_is_better=True)

model  = LogisticRegression(multi_class='auto',class_weight='balanced')
grid = RandomizedSearchCV(model, param_distributions=params, scoring = QWK_scorer,                     n_jobs=-1,cv=5,return_train_score=True) 
                                                
grid.fit(X_train_norm,y_train) 
print('time taken to train the model in sec:',time.time() - start)

model = grid.best_estimator_
model.fit(X_train_norm,y_train)


# In[489]:


actual_label = y_train
predicted_label = model.predict(X_train_norm)
print('Quadratic weighed kappa for training data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_cv
predicted_label = model.predict(X_cv_norm)
print('Quadratic weighed kappa for cross validation data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_test
predicted_label = model.predict(X_test_norm)
print('Quadratic weighed kappa for test data is :',calculate_QWK(actual_label,predicted_label))

plot_confusion_matrix(actual_label,predicted_label)


# ## Conclusion 
# * Even though we derived 19 new features still the linear model fails miserably.
# * QWK for test data is only 0.143 and almost half of the data misclassified.

# # Kernel SVM 
# * In this section we will use kernel support vector classifier. Since we are using kernels, we can drop those 19 new feature as kernel trick takes care of derived feature.

# ### feature standerdization

# In[495]:


# reference : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(X_train)
X_train_norm = normalizer.transform(X_train)
X_test_norm = normalizer.transform(X_test)
X_cv_norm = normalizer.transform(X_cv)


# In[497]:


# reference : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC

start = time.time()
params = {'C' : [0.0001,0.001,0.01,0.1,1,10],
          'kernel' : ['rbf','linear', 'poly', 'sigmoid']}

QWK_scorer = make_scorer(calculate_QWK, greater_is_better=True)

model  = SVC(gamma='scale',class_weight='balanced')
grid = RandomizedSearchCV(model, param_distributions=params, scoring = QWK_scorer,                     n_jobs=-1,cv=5,return_train_score=True) 
                                                
grid.fit(X_train_norm,y_train) 
print('time taken to train the model in sec:',time.time() - start)


# In[498]:


grid.best_estimator_


# In[500]:


# reference : https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.svm import SVC

start = time.time()
params = {'C' : [0.0001,0.001,0.01,0.1,1,10],
          'kernel' : ['rbf','linear', 'poly', 'sigmoid']}

QWK_scorer = make_scorer(calculate_QWK, greater_is_better=True)

model  = SVC(gamma='scale',class_weight='balanced')
grid = RandomizedSearchCV(model, param_distributions=params, scoring = QWK_scorer,                     n_jobs=-1,cv=5,return_train_score=True) 
                                                
grid.fit(X_train_norm,y_train) 
print('time taken to train the model in sec:',time.time() - start)

model = grid.best_estimator_
model.fit(X_train_norm,y_train)


# In[501]:


actual_label = y_train
predicted_label = model.predict(X_train_norm)
print('Quadratic weighed kappa for training data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_cv
predicted_label = model.predict(X_cv_norm)
print('Quadratic weighed kappa for cross validation data is :',calculate_QWK(actual_label,predicted_label))

actual_label = y_test
predicted_label = model.predict(X_test_norm)
print('Quadratic weighed kappa for test data is :',calculate_QWK(actual_label,predicted_label))

plot_confusion_matrix(actual_label,predicted_label)


# ## Conclusion
# * Even though we are using kernel trick, support vector classifier fails miserably to classify this data set.
# * QWK for test data is only 0.08 and 57% od data are misclassified.

# # Summary 

# In[551]:


from prettytable import PrettyTable

x = PrettyTable(['model','QWK_training','QWK_cv','QWK_test','misclassification(%)'])
x.add_row(['XgbClassifier',1.0,0.94,0.86,7.7])
x.add_row(['RandomForest',0.99,0.92,0.77,11.5])
x.add_row(['NeuralNetwork',0.71,0.70,0.34,33.1])
x.add_row(['LogisticRegression',0.80,0.80,0.14,50.8])
x.add_row(['SVC',0.70,0.68,0.08,57.09])
print(x)


# ### Best model is XgbClassifier with test QWK = 0.86 and only 7.7% misclassification

# # Key take away 
# * There is no hard and fast rule for selecting the model means we can not simply select a algorithm which performs best for all kind of data set.
# * The main motive to perform these many algorithms to see how each algorithm performs differenty from others.
# * In this data set tree based algorithms like XgbClassifier and RandomForest model performs extremely well where other linear model like SVC and LogisticRegression fails miserably. The reason behind this is linear model tends to perform well in high dimensional data as finding a linear hyperplan is easy in high dimension but here we have around 100 features only.
# * Small data set ( only 100 feature and only 17K data points ) limits further experiment. 
