#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from scipy.stats import randint 

from collections import Counter

from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj

init_notebook_mode(connected=True)
import plotly.figure_factory as ff



# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# To display all the columns
pd.options.display.max_columns = None

# To display all the rows
pd.options.display.max_rows = None

# To map Empty Strings or numpy.inf as Na Values
pd.options.mode.use_inf_as_na = True

pd.options.display.expand_frame_repr =  False

get_ipython().run_line_magic('matplotlib', 'inline')

# Set Style
sns.set(style = "whitegrid")


# In[19]:


patient_data = pd.read_csv('dataset.csv', low_memory = False, skipinitialspace = True, float_precision = 2)
patient_data.head()


# In[20]:


patient_data.shape


# In[21]:


# Check the total null values in each column.
print("Total NULL Values in each columns")
print("---------------------------------")
print(patient_data.isnull().sum())


# In[22]:


def data_label(ax, spacing = 5):

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}%".format(y_value)

        # Create annotation
        plt.annotate(
            label,                        # Use `label` as label
            (x_value, y_value),           # Place label at end of the bar
            xytext = (0, space),          # Vertically shift label by `space`
            textcoords = "offset points", # Interpret `xytext` as offset in points
            ha = 'center',                # Horizontally center label
            va = va)                      # Vertically align label differently for positive and negative values.


# In[23]:


# Univariate Plot Analysis of Ordered categorical variables vs Percentage Rate
categories = ['smoking', 'alcohol', 'diabetes', 'sex', 'covid_pos','covid_deaths']
counter = 1

plt.figure(figsize = (15, 12))

for col_list in categories:
        
    category_data = round(((patient_data[col_list].value_counts(dropna = False))/
                    (len(patient_data[col_list])) * 100), 2)

    plt.subplot(2, 3, counter)
    ax = sns.barplot(x = category_data.index, y = category_data.values, order = category_data.sort_index().index)
    plt.xlabel(col_list, labelpad = 15)
    plt.ylabel('Percentage Rate', labelpad = 10)

    # Call Custom Function
    data_label(ax)

    counter += 1

del categories, counter, ax

plt.subplots_adjust(hspace = 0.2)
plt.subplots_adjust(wspace = 0.6)
plt.show()


# In[24]:


colors = ['rgb(1, 1, 50)']

fig = ff.create_distplot(hist_data = [patient_data["age"].values], group_labels = ['age'], 
                         colors = colors, bin_size=3)

fig.update_layout(title_text='Distribution of Age')

fig.show()


# In[16]:


survived_cov = patient_data[patient_data["covid_deaths"] == 0]["age"]
not_survived_cov = patient_data[patient_data["covid_deaths"] == 1]["age"]


# In[25]:


colors = ['rgb(0, 0, 100)', 'rgb(0, 200, 200)']

hist_data = [survived_cov, not_survived_cov]
group_labels = ['Survived after covid', 'Not Survived after covid']

fig = ff.create_distplot(hist_data, group_labels, bin_size=[1, 1], colors = colors)
fig.update_layout(title_text = "Effect of Age on the Survival Rate")

fig.show()


# In[83]:


plt.figure(figsize=(10, 6))
sns.distplot(patient_data['oxygen_levels'], kde = True, color='red') #the line is called as a probability density function and it defines 
                                   # distribution of data
plt.show()


# In[84]:


hypertension = patient_data[patient_data['high_blood_pressure']==1]
no_hypertension = patient_data[patient_data['high_blood_pressure']==0]

hypertension_yes_survived = hypertension[patient_data["covid_deaths"] == 0]
hypertension_yes_not_survived = hypertension[patient_data["covid_deaths"] == 1]
hypertension_no_survived = no_hypertension[patient_data["covid_deaths"] == 0]
hypertension_no_not_survived = no_hypertension[patient_data["covid_deaths"] == 1]


# In[85]:


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                   subplot_titles=['Hypertension Distribution', 'Hypertension and Survival Rate'])

labels1 = ["Hypertension YES","Hypertension NO"]
values1 = [len(hypertension), len(no_hypertension)]

labels2 = ['Hypertension YES - Survived','Hypertension  YES - Not Survived', 'Hypertension NO - Survived',
           'Hypertension  NO - Not Survived']
values2 = [len(hypertension_yes_survived), len(hypertension_yes_not_survived),
          len(hypertension_no_survived), len(hypertension_no_not_survived)]

# fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.add_trace(go.Pie(labels=labels1, values=values1, name="Distribution of Hypertension"), 1, 1)
fig.add_trace(go.Pie(labels=labels2, values=values2, name='Hypertension & Survival'), 1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(title_text="Hypertension Distribution and Survival Rate...")

fig.show()


# In[26]:


cat = pd.melt(patient_data, id_vars=["previous_strokes"], value_vars=["cholesterol", "high_blood_pressure","cardio","active","covid_pos","covid_deaths"])


            # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
cat["total"]=1
cat=cat.groupby(["variable","previous_strokes","value"],as_index=False).count()


            # Draw the catplot with 'sns.catplot()'
dr=sns.catplot(x="variable", y="total", hue="value", data=cat,col="previous_strokes",kind = 'bar')

fig=dr.fig

fig.savefig('catplot.png')


# In[87]:


cat = pd.melt(patient_data, id_vars=["covid_deaths"], value_vars=["cough/fever", "high_blood_pressure"])


            # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
cat["total"]=1
cat=cat.groupby(["variable","covid_deaths","value"],as_index=False).count()


            # Draw the catplot with 'sns.catplot()'
dr=sns.catplot(x="variable", y="total", hue="value", data=cat,col="covid_deaths",kind = 'bar')

fig=dr.fig

fig.savefig('catplot.png')


# In[ ]:




