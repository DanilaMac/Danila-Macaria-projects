#!/usr/bin/env python
# coding: utf-8

# # Descriptive Analysis

# Predictive variables:
# - 'match_id' : volleyball match id
# - 'venue_id': stadium id were the volleyball match was played
# - 'league': volleyball match league
# - "classification": league classification (1: higher classification, 4: lower classification).
# - "country": country were the volleyball match is played
# - "continent": continent were the volleyball match is played
# - 'num_operators_day':  total number of operators per date 
# - 'date': match date
# - 'start_time' : time of match start
# - 'year': the year in which the match took place
# - 'months': the month in which the match took place
# - 'rank_week': week number (variables are 1 (first week), 2(second week), 3(third week), and 4 (fourth week)).
# - 'day': day of the month in which the match took place
# - 'weekday': day of the week in which the match took place
# - 'day_time': Grouped time ranges (early morning, morning, afternoon, night)
# - freq_rank_week_day_time: matches grouped per weekday and time             
# - freq_month_day_time: matches grouped per month and time             
# - 2h_interval: Match time duration                     
# - GLnum_matches_interval: total number of matches streamed by Genius Live Production from different disciplines per date and time        
# - Totalnum_matches_interval: total number of matches streamed by different  streaming productions from different disciplines per date and time        
# 
# Target variable:
# - 'issue': streaming problems (1= streaming problems, 0= no streaming problems)                  

# In[3]:


import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataprep.eda import create_report
#pip install dataprep
from dataprep.eda import create_report


# In[36]:


#Dataset imported
data = pd.read_csv('data_clean.csv',encoding='unicode_escape')
data.head(5)


# In[37]:


#column "Unnamed" deleted
data = data.drop(['Unnamed: 0'], axis=1)
data.head(5)


# # Dataprep report

# In[6]:


create_report(data)


# # First descriptive analysis from DataPrep Report 
# 
# - Most of the matches belong to Italian and French leagues.
# 
# - More than 70% of the matches belong to league classification 3 and 4, leagues of lower classification.
# 
# - Most of the matches were played at France and Portugal.
# 
# - 98% of the matches were played at Europe. 
# 
# - Most of the matches were played at 5, 4, 6 and 3 PM (16%, 12.39%, 11.08% and 10.89% respectively)
# 
# - Most of the data belong to matches played in 2021.
# 
# - 85% of the matches were played in January, February, March, October, November and December.
# 
# - Matches are equally distributed at weeks. Variable rank_week do not provide much information.
# 
# - 66% of the matches were played on Saturday and Sunday in the afternoon. 
# 
# - 82.40% of matches did not have issues during streaming. The target variable is unbalanced. 

# # Relationship between predictive and target variables

# In[38]:


fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20,20))
fig.suptitle('Análisis de variables')
for c, ax in zip(data.columns[:-1], axes.flatten()):
    sns.histplot(data = data.loc[data['issue']==0, c].dropna(), stat = 'density', ax = ax, kde = False )
    sns.histplot(data = data.loc[data['issue']==1, c].dropna(), stat = 'density', kde=False, ax=ax, color = 'lightblue')
    ax.legend(['Issue = 0', 'Issue = 1'])


# ### Matches per time interval and weekday

# In[39]:


plt.figure(figsize=(7, 5))
ax = sns.countplot(data=data, x='weekday', order=['Monday', "Tuesday", 'Wednesday', "Thursday",'Friday','Saturday','Sunday'],
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel('Weekday', size=14)
plt.yticks(size=12)
plt.ylabel('Number of matches', size=12)
plt.title("Matches per weekday", size=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - 66% of the matches were played on Saturday and Sunday. 54.3% of the matches were played on Saturday and Sunday and did not had issues during the streaming. 
# - 11.8% of the matches were played at Saturday and Sunday and had issues during the streaming. Most of the matches with issues were played on Saturday and Sunday

# In[40]:


# Matches grouped per weekday and time interval
Days_interval_issues = data.groupby(['2h_interval','weekday'])
Days_interval_issues_final= pd.DataFrame(Days_interval_issues["issue"].count())
Days_interval_issues_final= Days_interval_issues_final.unstack()

# Matches with issues grouped per weekday and time interval 
Days_interval_issues_2 = data.groupby(['2h_interval','weekday'])
Days_interval_issues_final_2= pd.DataFrame(Days_interval_issues_2["issue"].sum())
Days_interval_issues_final_2= Days_interval_issues_final_2.unstack()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(Days_interval_issues_final)
etiquetas = ['Friday', 'Monday', 'Saturday','sunday','thurday','Tuesday','Wednesday'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='Dias',frameon=True);
plt.ylabel('Number of matches',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches per weekday and time interval',fontsize = 20)

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(Days_interval_issues_final_2)
etiquetas = ['Friday', 'Monday', 'Saturday','sunday','thurday','Tuesday','Wednesday'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='Dias',frameon=True);
plt.ylabel('Number of matches with issues',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches with issues per weekday and time interval',fontsize = 20)

plt.show()
sns.set_theme()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(Days_interval_issues_final, annot=True, linewidths=.5, ax=ax)


# - Most of total matches and most matches with issues were played between 14 and 20 PM, with a maximum between 16 and 18 PM for matches played on weekend and with a maximum between 18 and 20 PM for matches that were played on Monday, Tuesday, Wednesday, Thursday, and Friday. Most of matches were played on Saturday and Sunday.

# ### Matches per time interval and week number

# In[41]:


plt.figure(figsize=(7, 5))
ax = sns.countplot(data=data, x='rank_week', order=[1, 2, 3, 4],
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel('Week number', size=14)
plt.yticks(size=12)
plt.ylabel('Number of matches', size=12)
plt.title("Matches per week number", size=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# In[10]:


# Matches grouped per week number and time interval
week_interval_issues_week_num = data.groupby(['2h_interval','rank_week'])
week_interval_issues_final_week_num= pd.DataFrame(week_interval_issues_week_num["issue"].count())
week_interval_issues_final_week_num= week_interval_issues_final_week_num.unstack()

# Matches with issues grouped per week number and time interval
week_interval_issues_2_week_num = data.groupby(['2h_interval','rank_week'])
week_interval_issues_final_2_week_num= pd.DataFrame(week_interval_issues_2_week_num["issue"].sum())
week_interval_issues_final_2_week_num= week_interval_issues_final_2_week_num.unstack()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(week_interval_issues_final_week_num)
etiquetas = ['1', '2', '3','4'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='WEEKS',frameon=True);
plt.ylabel('Number of matches',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches per week number and time interval',fontsize = 20)

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(week_interval_issues_final_2_week_num)
etiquetas = ['1', '2', '3','4'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='WEEKS',frameon=True);
plt.ylabel('Number of matches with issues',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches with issues per week number and time interval',fontsize = 20)

plt.show()
sns.set_theme()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(week_interval_issues_final_week_num, annot=True, linewidths=.5, ax=ax)


# - Total number of matches and matches with issues are equally distributed per week number. 
# - Total number of matches and number of matches with issues were played between 12 and 20 PM, with a maximum between 16 and 18 PM. 
# - The predictive variable "rank_week" do not provide much information, it will be dismissed in the future analysis.

# ### Matches per time interval and month

# In[42]:


data.months.value_counts()


# In[43]:


plt.figure(figsize=(7, 5))
ax = sns.countplot(data=data, x='months', order=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Sep", "Oct", "Nov", "Dec"],
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel("Mes", size=14)
plt.yticks(size=12)
plt.ylabel('Number of matches', size=12)
plt.title("Matches per month", size=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - 85% of matches were played in January, February, March, October, November and December (Most of the matches were played on Febraury and March). 14.1% of matches were played in January, February, March, October, November and December, and had issues during the streaming.
# - There were no matches played on July and August.

# In[44]:


# Matches grouped per month and time interval
month_interval_issues = data.groupby(['2h_interval','months'])
month_interval_issues_final= pd.DataFrame(month_interval_issues["issue"].count())
month_interval_issues_final_month= month_interval_issues_final.unstack()

# Matches with issues grouped per month and time interval
month_interval_issues_2 = data.groupby(['2h_interval','months'])
month_interval_issues_final_2= pd.DataFrame(month_interval_issues_2["issue"].sum())
month_interval_issues_final_2_month= month_interval_issues_final_2.unstack()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(month_interval_issues_final_month)
etiquetas = ['Apr','Dec','Feb','Jan','Jun','Mar','May','Nov','Oct','Sep'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='months',frameon=True);
plt.ylabel('Number of matches',fontsize=15)
plt.xlabel('Time inverval',fontsize=15)
ax.set_title("Matches per month and time interval",fontsize = 20)

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(month_interval_issues_final_2_month)
etiquetas = ['Apr','Dec','Feb','Jan','Jun','Mar','May','Nov','Oct','Sep'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='months',frameon=True);
plt.ylabel('Number of matches with issues',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches with issues per month and time interval',fontsize = 20)

plt.show()
sns.set_theme()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(month_interval_issues_final_month, annot=True, linewidths=.5, ax=ax)


# - Most of total matches and most matches with issues were played between 14 and 20 PM. There is no clear tendency on March and September. 

# ### Matches per league classification and time interval

# In[45]:


plt.figure(figsize=(7, 5))
ax = sns.countplot(data=data, x="classification", order=[1,2,3,4],
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel("League classification", size=14)
plt.yticks(size=12)
plt.ylabel('Number of matches', size=12)
plt.title("Matches per league classification", size=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - More than 70% of the matches belong to league classification 3 and 4, leagues of lower classification. 13% of the matches belong to classification 3 had issues during the streaming. 

# In[46]:


# Matches grouped per league classification and time interval
league_classification_interval_issues = data.groupby(['2h_interval','classification'])
league_classification_interval_issues_final= pd.DataFrame(league_classification_interval_issues["issue"].count())
league_classification_interval_issues_league= league_classification_interval_issues_final.unstack()

# Matches with issues grouped per league classification and time interval
league_classification_interval_issues_2 = data.groupby(['2h_interval','classification'])
league_classification_interval_issues_final_2= pd.DataFrame(league_classification_interval_issues_2["issue"].sum())
league_classification_interval_issues_2_league= league_classification_interval_issues_final_2.unstack()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(league_classification_interval_issues_league)
etiquetas = ['1','2','3','4'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='classification',frameon=True);
plt.ylabel('Number of matches',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches per league classification and time interval',fontsize = 20)

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(league_classification_interval_issues_2_league)
etiquetas = ['1','2','3','4'] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='league_clasification',frameon=True);
plt.ylabel('Number of matches with issues',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title("Matches with issues per league classification and time interval",fontsize = 20)

plt.show()
sns.set_theme()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(league_classification_interval_issues_league, annot=True, linewidths=.5, ax=ax)


# - Most of matches of all leagues were played between 14 and 20 PM, with a maximum between 16 and 18 PM for matches that belong to league 3.

# ### Matches per country

# In[47]:


data.country.value_counts()


# In[48]:


data.country.unique()


# In[49]:


plt.figure(figsize=(7, 5))
ax = sns.countplot(data=data, x='country', order=['Albania','Austria','Belgium','Bulgaria','Czech Republic','Denmark','Finland','France','Iceland','Italy','New Zealand','Portugal','Slovakia','Slovenia','Sweden','Ukraine'],
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel("País", size=14)
plt.yticks(size=12)
plt.ylabel('Number of matches', size=12)
plt.title("Matches per country", size=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - 26% of the matches were played at France and Portugal and 4.5% of the matches were played at these countries and had issues during the streaming.

# In[50]:


#Creaton of masks for matches with issues and matches country
mask_issue = data["issue"] == 1
mask_france = data["country"] == "France"
mask_portugal = data["country"] == "Portugal"
mask_ukraine = data["country"] == "Ukraine"
mask_italy = data["country"] == "Italy"
mask_finland = data["country"] == "Finland"
mask_austria = data["country"] == "Austria"
mask_sweden = data["country"] == "Sweden"
mask_slovenia = data["country"] == "Slovenia"
mask_denmark = data["country"] == "Denmark"
mask_slovakia = data["country"] == "Slovakia"
mask_iceland = data["country"] == "iceland"
mask_CzechRepublic = data["country"] == "Czech Republic"
mask_Belgium = data["country"] == "Belgium"
mask_NewZealand  = data["country"] == "New Zealand"
mask_Bulgaria   = data["country"] == "Bulgaria"
mask_Albania   = data["country"] == "Albania"

df_mask_franceissue = data.loc[mask_issue & mask_france]
df_mask_portugalissue = data.loc[mask_issue & mask_portugal]
df_mask_ukrainelissue = data.loc[mask_issue & mask_ukraine]
df_mask_italyissue = data.loc[mask_issue & mask_italy]
df_mask_finlandissue = data.loc[mask_issue & mask_finland]
df_mask_austriaissue = data.loc[mask_issue & mask_austria]
df_mask_swedenissue = data.loc[mask_issue & mask_sweden]
df_mask_sloveniaissue = data.loc[mask_issue & mask_slovenia]
df_mask_denmarkissue = data.loc[mask_issue & mask_denmark]
df_mask_slovakiaissue = data.loc[mask_issue & mask_slovakia]
df_mask_icelandissue = data.loc[mask_issue & mask_iceland]
df_mask_CzechRepublicissue = data.loc[mask_issue & mask_CzechRepublic]
df_mask_Belgiumissue = data.loc[mask_issue & mask_Belgium]
df_mask_NewZealandissue = data.loc[mask_issue & mask_NewZealand]
df_mask_Bulgariaissue = data.loc[mask_issue & mask_Bulgaria]
df_mask_Albaniaissue = data.loc[mask_issue & mask_Albania]


df_franceissue = data.loc[mask_france]
df_portugalissue = data.loc[mask_portugal]
df_ukrainelissue = data.loc[mask_ukraine]
df_italyissue = data.loc[mask_italy]
df_finlandissue = data.loc[mask_finland]
df_austriaissue = data.loc[mask_austria]
df_swedenissue = data.loc[mask_sweden]
df_sloveniaissue = data.loc[mask_slovenia]
df_denmarkissue = data.loc[mask_denmark]
df_slovakiaissue = data.loc[mask_slovakia]
df_icelandissue = data.loc[mask_iceland]
df_CzechRepublicissue = data.loc[mask_CzechRepublic]
df_Belgiumissue = data.loc[mask_Belgium]
df_NewZealandissue = data.loc[mask_NewZealand]
df_Bulgariaissue = data.loc[mask_Bulgaria]
df_Albaniaissue = data.loc[mask_Albania]

print("matches with issues per country")

print(len(df_mask_franceissue.index))
print(len(df_mask_portugalissue.index))
print(len(df_mask_ukrainelissue.index))
print(len(df_mask_italyissue.index))
print(len(df_mask_finlandissue.index))
print(len(df_mask_austriaissue.index))
print(len(df_mask_swedenissue.index))
print(len(df_mask_sloveniaissue.index))
print(len(df_mask_denmarkissue.index))
print(len(df_mask_slovakiaissue.index))
print(len(df_mask_icelandissue.index))
print(len(df_mask_CzechRepublicissue.index))
print(len(df_mask_Belgiumissue.index))
print(len(df_mask_NewZealandissue.index))
print(len(df_mask_Bulgariaissue.index))
print(len(df_mask_Albaniaissue.index))

#Albania and Iceland do not have matches with issues


# In[18]:


#percentage of matches with issues country
print("percentage of matches with issues France:", len(df_mask_franceissue.index)*100/(len(df_franceissue.index)))
print("percentage of matches with issues Portugal:",len(df_mask_portugalissue.index)*100/(len(df_portugalissue.index)))
print("percentage of matches with issues Ukraine:",len(df_mask_ukrainelissue.index)*100/(len(df_ukrainelissue.index)))
print("percentage of matches with issues Italy:",len(df_mask_italyissue.index)*100/(len(df_italyissue.index)))
print("percentage of matches with issues Finland:",len(df_mask_finlandissue.index)*100/(len(df_finlandissue.index)))
print("percentage of matches with issues Austria:",len(df_mask_austriaissue.index)*100/(len(df_austriaissue.index)))
print("percentage of matches with issues Sweden:",len(df_mask_swedenissue.index)*100/(len(df_swedenissue.index)))
print("percentage of matches with issues Slovenia:",len(df_mask_sloveniaissue.index)*100/(len(df_sloveniaissue.index)))
print("percentage of matches with issues Denmark:",len(df_mask_denmarkissue.index)*100/(len(df_denmarkissue.index)))
print("percentage of matches with issues Slovakia:",len(df_mask_slovakiaissue.index)*100/(len(df_slovakiaissue.index)))
print("percentage of matches with issues Czech Republic:",len(df_mask_CzechRepublicissue.index)*100/(len(df_CzechRepublicissue.index)))
print("percentage of matches with issues Belgium:",len(df_mask_Belgiumissue.index)*100/(len(df_Belgiumissue.index)))
print("percentage of matches with issues New Zealand:",len(df_mask_NewZealandissue.index)*100/(len(df_NewZealandissue.index)))
print("percentage of matches with issues Bulgaria:",len(df_mask_Bulgariaissue.index)*100/(len(df_Bulgariaissue.index)))


# In[51]:


issues_country = {'France': 16.90909090909091, 'Portugal':18.06451612903226 , 'Ukraine':16.167664670658684, 'Italy':23.746312684365783, "Finland":14.781297134238311, "Austria": 18.047882136279927, "Sweden":13.003663003663004, "Slovenia":17.418032786885245, "Denmark":11.859838274932615, "Slovakia":16.976744186046513, "Czech Republic":16.78082191780822, "New Zeland":45.833333333333336, "Belgium": 20.21, "Bulgaria": 20.0, "Albania": 0.0, "Iceland": 0.0}     
country_list = list(issues_country.keys())
porcentaje_issues_country = list(issues_country.values())
fig = plt.figure(figsize = (16.6, 5))
plt.bar(country_list, porcentaje_issues_country, color ='lightblue',
        width = 0.4)
 
plt.xlabel("Country")
plt.ylabel("Percentage of matches with issues")
plt.title("Percentage of matches with issues per country")
plt.show()


# - New Zealand is the country with higher percentage of matches with issues.

# In[52]:


# Matches grouped per country and time interval
country_interval_issues = data.groupby(['2h_interval','country'])
country_interval_issues_final= pd.DataFrame(country_interval_issues["issue"].count())
country_interval_issues_final= country_interval_issues_final.unstack()

# Matches with issues grouped per country and time interval 
country_interval_issues_2 = data.groupby(['2h_interval','country'])
country_interval_issues_final_2= pd.DataFrame(country_interval_issues_2["issue"].sum())
country_interval_issues_final_2= country_interval_issues_final_2.unstack()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(country_interval_issues_final)
etiquetas = ['Albania','Austria','Belgium','Bulgaria','Czech Republic','Denmark','Finland','France','Iceland','Italy','New Zealand','Portugal','Slovakia','Slovenia','Sweden','Ukraine', "Belgium", "Bulgaria", "Iceland"] 
ax.legend(etiquetas, loc='best',fontsize='medium',title='country',frameon=True);
plt.ylabel('Number of matches',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title("Matches per country and time interval",fontsize = 20)

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(country_interval_issues_final_2)
etiquetas = ['Albania','Austria','Belgium','Bulgaria','Czech Republic','Denmark','Finland','France','Iceland','Italy','New Zealand','Portugal','Slovakia','Slovenia','Sweden','Ukraine',"Belgium", "Bulgaria", "Iceland"]
ax.legend(etiquetas, loc='best',fontsize='medium',title='country',frameon=True);
plt.ylabel('Number of matches with issues',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title("Matches with issues per country and time interval",fontsize = 20)

plt.show()
sns.set_theme()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(country_interval_issues_final, annot=True, linewidths=.5, ax=ax)


# - Most of total matches and matches with issues played in France ocurred between 18 and 20 PM. Rest of total matches and matches with issues were played between 14 and 20 PM.

# ### Anlysis of variable "GLnum_matches_interval":
# Volleyball matches grouped per date and time interval streamed by Genius Live Production

# In[53]:


# Matches grouped per date and time interval
data_real_concurrency = data.groupby(['date','2h_interval']).league.count()
data_real_concurrency


# In[54]:


# merge variable data_real_concurrency with original dataset
data = pd.merge(data, data_real_concurrency, on=['date','2h_interval']) 

data.head(4)


# In[55]:


# GLnum_matches_interval original variable will not be used. Variable dropped.
data = data.drop(['GLnum_matches_interval'],axis=1)


# In[56]:


# We named new variable created "league_y" as "GLnum_matches_interval" --> Volleyball matches grouped per date and time interval
data.rename(columns={'league_y':'GLnum_matches_interval'}, inplace=True)


# In[57]:


data[data.GLnum_matches_interval > 37].head(40)


# In[58]:


data.GLnum_matches_interval.value_counts()


# In[59]:


plt.figure(figsize=(15, 5))
ax = sns.countplot(data=data, x='GLnum_matches_interval', 
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel("Total number of Volleyball GL matches per day and time interval", size=14)
plt.yticks(size=12)
plt.ylabel('Number of Volleyball GL matches', size=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - Most of total matches and matches with issues were streamed in parallel with few matches.

# ### Matches per year

# In[60]:


plt.figure(figsize=(15, 5))
ax = sns.countplot(data=data, x='year', 
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel("Year", size=14)
plt.yticks(size=12)
plt.ylabel('Number of matches', size=12)
plt.title("Matches per year", size=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - Most of matches and matches with issues were played at 2021.

# ### Analysis of variable "Totalnum_matches_interval":
# (total number of matches streamed by different streaming productions from different disciplines per date and time)

# In[61]:


plt.figure(figsize=(25, 7))
ax = sns.countplot(data=data, x='Totalnum_matches_interval', 
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel("Number of matches streamed by different  streaming productions from different disciplines per date and time", size=14)
plt.yticks(size=12)
plt.ylabel('Number of Volleyball matches streamed by Genius Live', size=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - There is no clear tendency about this variable.

# ### Matches per number of daily operators

# In[62]:


plt.figure(figsize=(10, 5))
ax = sns.countplot(data=data, x='num_operators_day', 
                   hue='issue', hue_order=[0,1],
                   palette=['lightblue', 'violet'])
plt.xticks(size=12)
plt.xlabel("Number of daily operators", size=14)
plt.yticks(size=12)
plt.ylabel('Número de partidos', size=12)
plt.title("Matches per number of daily operators", size=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

total = len(data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')
plt.tight_layout()
plt.show()


# - Most of total matches and most of matches with issues were streamed with 4 to 7 operators.

# In[63]:


# Matches grouped per number of daily operators and time interval
numop_interval = data.groupby(['2h_interval','num_operators_day'])
numop_interval_final= pd.DataFrame(numop_interval["issue"].sum())
numop_interval_final= numop_interval_final.unstack()

# Matches with issues grouped per number of daily operators and time interval
numop_interval_2 = data.groupby(['2h_interval','num_operators_day'])
numop_interval_final_2= pd.DataFrame(numop_interval_2["issue"].sum())
numop_interval_final_2= numop_interval_final_2.unstack()

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(numop_interval_final)
etiquetas = ['2','3','4','5','6','7','8','9','10','11','13','14'] 
ax.legend(etiquetas, loc='best',fontsize='small',title='Daily operators',frameon=True);
plt.ylabel('Number of matches',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches per daily operators and time interval',fontsize = 20)

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(15,3),dpi=400)
ax = plt.axes()
ax.plot(numop_interval_final_2)
etiquetas = ['2','3','4','5','6','7','8','9','10','11','13','14'] 
ax.legend(etiquetas, loc='best',fontsize='small',title='Daily operators',frameon=True);
plt.ylabel('Number of matches with issues',fontsize=15)
plt.xlabel('Time interval',fontsize=15)
ax.set_title('Matches per daily operators and time interval',fontsize = 20)


# - Most of total matches and most of matches with issues were played between 14 and 20 PM for 4,5,6,7 and 8 daily operators.
# - For the rest of matches with 2,3,9,10,11,13 and 14 daily operators, there is not clear tendency.

# #### Final dataset

# In[64]:


data.columns


# In[34]:


# columns "match_id" and "venue_id" deleted
data = data.drop(['match_id', 'venue_id'], axis=1)
data.head()


# In[65]:


# Reordenar las columnas 
data = data[['league_x','classification',
             'country','continent','num_operators_day', 'date','start_time','year',
             'months','rank_week','day', 'weekday', 'day_time', 'freq_weekday_day_time','freq_rank_week_day_time','freq_month_day_time',
             '2h_interval', 'GLnum_matches_interval', 'Totalnum_matches_interval','issue']]


# In[66]:


#columns renamed
data = data.rename({'league_x':'league','classification':'league_level', '2h_interval': 'start_time_interval', 'Totalnum_matches_interval':'Totlnum_match_y_m_wd_int'}, axis=1)


# In[38]:


data.info()


# ### Final columns
# 
# Predictive variables:
# 
# - 'league': volleyball match league 
# - "league_level": league classification (1: higher classification, 4: lower classification).
# - "country": country were the volleyball match is played
# - "continent": continent were the volleyball match is played
# - 'num_operators_day': total number of operators per date
# - 'date': match date
# - 'start_time' : time of match start
# - 'year': the year in which the match took place
# - 'months': the month in which the match took place
# - 'rank_week': week number (variables are 1 (first week), 2(second week), 3(third week), and 4 (fourth week)).
# - 'day': day of the month in which the match took place
# - 'weekday': day of the week in which the match took place
# - 'day_time': Grouped time ranges (early morning, morning, afternoon, night)
# - freq_rank_week_day_time: matches grouped per weekday and time
# - freq_month_day_time: matches grouped per month and time
# - 2h_interval: Match time duration
# - GLnum_matches_interval:  Volleyball matches grouped per date and time interval
# - Totalnum_matches_interval: total number of matches streamed by different streaming productions from different disciplines per date and time
# 
# Target variable:
# 
# - 'issue': streaming problems (1= streaming problems, 0= no streaming problems)

# In[67]:


# Save dataset as cvs 
data.to_csv('data_4_models.csv')


# In[ ]:




