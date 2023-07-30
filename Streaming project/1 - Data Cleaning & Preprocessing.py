#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning & Preprocessing

# ### Theoretical framework
# 
# The dataset used in this work belongs to "Genius Live Production", a streaming company of sport matches. Clients has to download a mobile application to watch the stream. The aim of this work is to analyze the reasons of failures (issues), in transmissions during streamings, in order to find solutions to this problem.
# 
# ### Dataset description
# 
# Data contains information of 8000 Volleyball matches that occured between September 2020 and June 2022. 
# Columns:
# 
# - "date": match date
# - "time": time of match start
# - "interval": Match time duration (2 hours interval)
# - "GS_id": match Id
# - "id": match id
# - "fixture name": sport teams that compete
# - "continent": continent were the match is played
# - "country": country were the match is played
# - "league_id": id of league 
# - "league": match league 
# - "state": Actual state of match streaming. "Aborted": transmission cancelled before match start; "Confirm": match ready to stream, "Completed": match ended without video recording, "VOD": match ended and with video recording.
# - "issue_origin": origin of transmission issue.
# - "issue_severity": severity of issue transmission
# - "description": description of issue transmission
# - "venue_id": stadium id were the match was played
# - "venue_name": stadium name were the match is played
# 
# 
# ### External datasets
# 
# 3 external datasets from "Genius Live Production", were also used to get further information:
# 
# - 'numpartidos.csv': contains information of other sport matches streamed at the same time that volleyball matches.
# 
# - 'league_importance.csv': detailed league ranking
# 
# - 'data_operators.csv': contains number of stream operators per date

# In[1]:


import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# In[2]:


#Dataset imported
data = pd.read_csv('streamdata.csv',encoding='unicode_escape')
data


# In[3]:


#Columns of the dataset
data.columns


# In[4]:


#search for null values in each column.
data.isnull().sum() / data.shape[0]
#The columns "continent" and "country" contain missing values.


# In[5]:


#search for duplicated matches
id_duplicates = data.duplicated(subset= ["id"])
id_duplicates.value_counts()
#there are no duplicated matches


# ### Analysis of "state" variable

# In[6]:


data.state.value_counts()


# In[7]:


# "Confirmed" values deleted, since they provide very little information 
data.drop(data.index[data['state'] == "Confirmed"], inplace = True)
data.state.value_counts()


# In[8]:


# "Aborted" values deleted. Transmission of "Aborted" matches was cancelled before match start and this analysis 
# is focus on matches that were not televised due to failures in transmision. 
data.drop(data.index[data['state'] == "Aborted"], inplace = True)
data.state.value_counts()


# ###  Analysis of "continent" variable

# In[9]:


data.continent.value_counts()


# In[10]:


# Matches with null values at "continent" column
datacont = data[data.continent.isnull()]
datacont.head(4)


# In[11]:


# Leagues of matches with null values at "continent" column
datacont.league.unique()
# Leagues belong to Europe continent


# In[12]:


# Null values at "continent" column were replaced with "Europe" 
datacont.continent = 'Europe'
datacont.head()


# In[13]:


maskindex =  datacont.index


# In[14]:


# Matches with null values replaced added to original dataset
data.continent[maskindex] = datacont.continent[maskindex]


# In[15]:


print(data.continent.isnull().sum())


# ### Analysis of "country" variable

# In[16]:


# Matches per country
# 133 matches have "Europe" as country.
data_grouped = data.groupby('country')
type(data_grouped)
print(data_grouped.size())


# In[17]:


# Number of matches with null values at "Country" column
print(data.loc[data.country.isnull(), ].shape[0])


# In[18]:


# Selection of matches with null values at "Country" column
mask = data.country.isnull()
data_mask = data[mask]


# In[19]:


# Leagues of matches with null values at "Country" column
data_mask[["league"]].value_counts()


# In[20]:


# Use of regular expressions to obtain country name of Leagues from matches with null values at "Country" column

country_pattern = "(?P<country>Ukraine|Czech Republic|Portugal|Iceland|Sweden|Albania)"


country_regex = re.compile(country_pattern)


data_league_regular = data_mask.league.apply(str)


result_league = data_league_regular.apply(lambda x: country_regex.search(x))

print(result_league.isnull().sum())

print(result_league.notnull().sum())

num_country = result_league.apply(lambda x: x if x is None else x.group("country"))


# In[21]:


# Null values at "country" column were replaced with countries obtained with regular expressions 
data_mask.country = num_country


# In[22]:


# Matches with null values replaced added to original dataset
data.country[data_mask.index] = data_mask.country[data_mask.index]


# In[23]:


# 6 matches belong to league "[Test] Volleyball", country name could not be obtained with regular expressions.
# 6 matches with missing information will be deleted.
data.info()


# In[24]:


# Matches that have "Europe" as country
mask_data_country_europe = data.country == "Europe"
data_europe = data.loc[mask_data_country_europe]


# In[25]:


data_europe


# In[26]:


#venue id (stadium id were the match was played), of matches that have "Europe" as country
data_europe["venue_id"].value_counts()


# In[27]:


#matches that do not have "Europe" as country
mask_data_country = data.country != "Europe"
data_country = data.loc[mask_data_country] 


# In[28]:


#matches with the respective venue ids were played at Austria
mask_venue_id_129194 = data_country.venue_id == "129194"   #Austria 
mask_venue_id_130741 = data_country.venue_id == "130741"   #Austria  
mask_venue_id_130598 = data_country.venue_id == "130598"   #Austria
mask_venue_id_Austria = mask_venue_id_129194| mask_venue_id_130741|mask_venue_id_130598
data_country = data_country.loc[mask_venue_id_Austria]
data_country.head(20)


# In[29]:


#"Austria" was assigned at matches with "Europe" values at Country column, from the information obtained from venue id
data_mask_129194 = data.venue_id == "129194"
data_mask_130741 = data.venue_id == "130741"
data_mask_130598 = data.venue_id == "130598"
data_mask_129207 = data.venue_id == "129207"
data_mask_Austria = data_mask_129194 |data_mask_130741|data_mask_130598|data_mask_129207
mask_index_Austria = data.loc[data_mask_Austria].index
data.country[mask_index_Austria] = "Austria"


# In[30]:


#matches with the respective venue ids were played at Slovenia
mask_venue_id_129129 = data_country.venue_id == "129129"   #Slovenia
mask_venue_id_129116 = data_country.venue_id == "129116"   #Slovenia 
mask_venue_id_138203 = data_country.venue_id == "138203"   #Slovenia
mask_venue_id_129181 = data_country.venue_id == "129181"   #Slovenia
mask_venue_id_Slovenia = mask_venue_id_129129 |mask_venue_id_129116|mask_venue_id_138203|mask_venue_id_129181
data_country = data_country.loc[mask_venue_id_Slovenia]
data_country


# In[31]:


#"Slovenia" was assigned at matches with "Europe" values at Country column, from the information obtained from venue id
data_mask_129129 = data.venue_id == "129129"
data_mask_129116 = data.venue_id == "129116"
data_mask_138203 = data.venue_id == "138203"
data_mask_129181 = data.venue_id == "129181"
data_mask_Slovenia = data_mask_129129 |data_mask_129116|data_mask_138203|data_mask_129181
mask_index_Slovenia = data.loc[data_mask_Slovenia].index
data.country[mask_index_Slovenia] = "Slovenia"


# In[32]:


#matches with the respective venue ids were played at Slovakia
mask_venue_id_138450 = data_country.venue_id == "138450"   #slovaquia
mask_venue_id_137618 = data_country.venue_id == "137618"   #slovaquia  
mask_venus_id_Slovakia = mask_venue_id_138450|mask_venue_id_137618
data_country = data_country.loc[mask_venus_id_Slovakia]
data_country


# In[33]:


#"Slovakia" was assigned at matches with "Europe" values at Country column, from the information obtained from venue id
data_mask_138450 = data.venue_id == "138450"
data_mask_137618 = data.venue_id == "137618"
data_mask_129142 = data.venue_id == "129142"
data_mask_Slovakia = data_mask_138450 |data_mask_137618 | data_mask_129142
mask_index_Slovakia = data.loc[data_mask_Slovakia].index
data.country[mask_index_Slovakia] = "Slovakia"


# In[34]:


data["country"].value_counts()


# In[35]:


#7 matches with "Europe" as country do not have information at venue id
mask_data_pais_europa = data.country == "Europe"
data_europa = data.loc[mask_data_pais_europa]
data_europa["venue_id"].value_counts()


# In[36]:


#7 matches with "Europe" as country dropped
mask_europe =data.loc[data['country']=='Europe'].index
data.drop(mask_europe, inplace=True)


# In[37]:


data["country"].value_counts()


# ### Analysis of "league" variable

# In[38]:


#Matches that belongs to league "[Test] Volleyball"
sum(data['league'] == "[Test] Volleyball")


# In[39]:


data.shape


# In[40]:


#Matches are incorrectly classified as "[Test] Volleyball, the league does not exist
#Matches that belongs to league "[Test] Volleyball" dropped 
data.drop(data.index[data['league'] == "[Test] Volleyball"], inplace = True)
data.shape


# ### Analysis of "venue_id" variable

# In[41]:


data.venue_id.unique()


# In[42]:


#matches with "#VALUE" at venue_id  columnd were dropped
print(data[data.venue_id == '#VALUE!'].shape)
data[data.venue_id == '#VALUE!'].head()


# In[43]:


data.drop(data.index[data.venue_id == '#VALUE!'], inplace = True)
data.shape


# In[44]:


#"vanue_id" column transformed into "int" format
data = data.astype({"venue_id": "int64" })
#"venue_id" value replaced 
data = data.replace({'venue_id': {0: 1}}) 


# # Creation of new variables

# ### Analysis of "date" variable

# In[45]:


#"Date" variable converted into datetime variable at new variable "date_conv":
data['date_conv'] = pd.to_datetime(data.date, format= '%m/%d/%Y' )
print(data.dtypes)
data.head(5)


# In[46]:


#Values sorted by ascending "date_conv"
data.sort_values('date_conv', axis=0,ascending=False)


# ### Creation of variables: "year", "month", "day", "weekday" and "rank_week"

# In[47]:


#Year, month and day extracted from "date_conv" variable
#"Year", "month" and "day" variables created
data['year'] = data['date_conv'].dt.year 
data['month'] = data['date_conv'].dt.month
data['day'] = data['date_conv'].dt.day
data.head(5)


# In[48]:


#matches per year
print(data.year.value_counts())


# In[49]:


#matches per month
data.month.value_counts()


# In[50]:


#Month values renamed at "months" new variable
months   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Ago', 'Sep', 'Oct', 'Nov', 'Dec']

data['months'] = data.month.apply(lambda x: months[x-1])
data.months.value_counts()
#there are no matches at August


# In[51]:


#Few matches belong to July, this is consider a non representative sample
#Matches played at July were dropped 
data.drop(data.index[data['months'] == "Jul"], inplace = True)
data.shape


# In[52]:


#"weekday" new variable created from "date_conv" variable
week   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

data['weekday'] = data.date_conv.apply(lambda x: week[x.weekday()])
data


# In[53]:


#matches per day
print(data.weekday.value_counts())


# In[54]:


#"rank_week" (week number) new variable created from "day" variable.
#"rank_week" values are 1, 2, 3 and 4 (weekly rank)
data['rank_week'] = data.apply(lambda x: 1 if (x['day']<= 8) else (2 if ( 8 < x['day'] <=16) else (3 if (16 < x['day'] <=23) else  4)), axis=1)
data


# In[55]:


#date separated per time
print(data.time.dtype)
print(data.time.value_counts)


# ### Creation of new variables: "time_24" and "time_24_range"

# Format of "time" variable will be converted to 24-hour notation.
#  New column created (time_y), with "AM"/"PM" information.
#  Original "time" column renamed as "time_x"

# In[56]:


time_pattern = "(?P<franja>AM|PM)"
time_regex = re.compile(time_pattern)
result2 = data.time.apply(lambda x: time_regex.search(x))
print(result2.head(3))


result3 = result2.apply(lambda x: x[0])
print(result3.head(3))


data =data.merge(result3,left_index=True, right_index=True)
data.head(3)


# In[57]:


#"AM"/"PM" information deleted from "time_x" column
series_time = data.time_x

time_pattern3 = "(?P<franja>AM|PM)"

time_regex3 = re.compile(time_pattern3)

result4 = series_time.apply(lambda x: time_regex3.sub('', x))
print(result4.head(3))


# In[58]:


#information merged
data =data.merge(result4,left_index=True, right_index=True)

#Original "time_x" column renamed as "time_x_x"
#New column "time_x_y" with time without AM/PM information
#Column time_y contains "AM"/"PM" information

print(data.head(3))


# In[59]:


#Spaces at column time_x_y deleted
series_time_final = data.time_x_y

time_pattern_final = "\s"

time_regex_final = re.compile(time_pattern_final)

result_final = series_time_final.apply(lambda x: time_regex_final.sub('', x))
print(result_final.head(3))

data =data.merge(result_final,left_index=True, right_index=True)
data.head()
#Original "time_x_y" column renamed as "time_x_y_x"
#New column with "time_x_y" information and without spaces, named as "time_x_y_y"


# In[60]:


#":" pattern deleted from column "time_x_y_y"
series_time_final2 = data.time_x_y_y

time_pattern_final2 = ":"

time_regex_final2 = re.compile(time_pattern_final2)

result_final2 = series_time_final2.apply(lambda x: time_regex_final2.sub('', x))
print(result_final2)

data = data.merge(result_final2,left_index=True, right_index=True)
data.head()
#Original "time_x_y_y" column renamed as "time_x_y_y_x"
#New column with "time_x_y_y" information and without ":" pattern, named as "time_x_y_y_y"


# In[61]:


#Column "time_x_y_y_y" transformed to "int" format
data.time_x_y_y_y = data.time_x_y_y_y.apply(int)

print(data.time_x_y_y_y.dtype)

print(data.time_x_y_y_y.unique)


# Time converted to 24-hour notation at new column "time_24":
# - 12 hours substracted from rows of column "time_x_y_y_y" that belong to "AM" time range and are between 12:00 and 12:59 (120000 y 125900 respectively in "int" format).
# - 12 hours added to rows of column "time_x_y_y_y" that belong to "PM" time range and are before 12:00 (120000 in "int" format).

# In[62]:


data["time_24"] = data.apply(lambda x: (x['time_x_y_y_y'] - 120000) if (120000 <= x['time_x_y_y_y'] <= 125900) and x['time_y'] == "AM" else ((x['time_x_y_y_y'] + 120000) if x['time_x_y_y_y'] < 120000 and x['time_y'] == "PM" else x['time_x_y_y_y']), axis=1)

data["time_24"] = data["time_24"] / 100

data.head(3)


# Grouped time ranges at column "time_24_range":
# - early morning: 00:00 am to 5.59 am (6 hours)
# - morning: 6:00  am 11:59 am (6 hours)
# - afternoon: 12:00 pm a 5:59 pm (6 hours)
# - night: 6:00 pm a 11:59 pm (6 hours)

# In[63]:


data["time_24_range"] = data.apply(lambda x: "early morning" if (0 <= x['time_24'] <= 559)  else ("morning" if (600 <= x['time_24'] <= 1159) else ("afternoon" if (1200 <= x['time_24'] <= 1759)  else "night")), axis=1)
print(data.time_24_range.value_counts())

data.head(3)


# In[64]:


#Deletion of unnecessary columns
data = data.drop(['time_y','time_x_y_x','time_x_y_y_x','time_x_y_y_y','time_24'], axis=1)
#conservamos la columna "time_24_range"
data.columns


# ### Analysis of "issue" variable

# In[65]:


#"issue" variable transformed into binary format, 1: streaming problems, 0: no streaming problems

data['issue'] = data.apply(lambda x: 0 if x['issue_origin'] == "No issue related" else 1, axis=1)
data.head(3)


# ### Creation of new variables: 'scoreboard_issue', 'internet_issues', 'schedule_issue', 'quality_issue', 'phone_issue', 'extracamera_issue', 'vod_issue','poc_issue' 

# In[66]:


#New dataset created with 'description','issue_origin','issue_severity' variables 
data_description= data[['description','issue_origin','issue_severity']]
data_description = data_description.applymap(lambda x: x if np.isreal(x) else str(x).lower())


# In[67]:


data_description


# In[68]:


#use of regular expressions for search for pattern:
pattern = "(?P<scoreboard>scoreboard|overlay|score|escoresheet|stuck|display|data)"
regex = re.compile(pattern)


# In[69]:


#creation of new binary variable "scoreboard_issue" (1 = issue, 0 = no issue)
scoreboard = data_description.description.apply(lambda x: regex.search(x)) 
scoreboardClean = scoreboard.apply(lambda x: 0 if x is None else 1) 
data_description["scoreboard_issue"] = scoreboardClean 
data_description.scoreboard_issue.value_counts()


# In[70]:


#histogram of 'scoreboard_issue' variable
sns.histplot(data=data_description, x= 'scoreboard_issue',stat='count', shrink=0.1,  discrete = True)


# In[71]:


#creation of new binary variable "internet_issues" (1 = issue, 0 = no issue)
patron = "(?P<internet>bandwidth|internet|unstable|onnection|connectivity|signal|drop|feed|lost|outage|wifi|network|connected|offline|electricity|firewall)"
regex = re.compile(patron)
Internet = data_description.description.apply(lambda x: regex.search(x)) 
InternetClean = Internet.apply(lambda x: 0 if x is None else 1) 
data_description["internet_issues"] = InternetClean 
data_description.internet_issues.value_counts()


# In[72]:


#histogram of 'internet_issues' variable
sns.histplot(data=data_description, x= 'internet_issues',stat='count', shrink=0.1,  discrete = True)


# In[73]:


#creation of new binary variable "schedule_issue" (1 = issue, 0 = no issue)
patron = "(?P<schedule>assign|incorrect|schedule|back-to-back|not create|mapping)"
regex = re.compile(patron)
schedule = data_description.description.apply(lambda x: regex.search(x)) 
scheduleClean = schedule.apply(lambda x: 0 if x is None else 1)
data_description["schedule_issue"] = scheduleClean 
data_description.schedule_issue.value_counts()


# In[74]:


#histogram of 'schedule_issue' variable
sns.histplot(data=data_description, x= 'internet_issues',stat='count', shrink=0.1,  discrete = True)


# In[75]:


#creation of new binary variable "quality_issue" (1 = issue, 0 = no issue)
patron = "(?P<quality>bad view|too close|quality|noise|poor|artifacting|artefacting|visible|freez|court|early|sound|view|unwatch|wave|waving|case|casing|shak|focus|buffer|phone place)"
regex = re.compile(patron)
quality = data_description.description.apply(lambda x: regex.search(x))  
qualityClean = quality.apply(lambda x: 0 if x is None else 1) 
data_description["quality_issue"] = qualityClean 
data_description.quality_issue.value_counts()


# In[76]:


#histogram of 'quality_issue' variable
sns.histplot(data=data_description, x= 'quality_issue',stat='count', shrink=0.1,  discrete = True)


# In[77]:


#creation of new binary variable "phone_issue" (1 = issue, 0 = no issue)
patron = "(?P<phone>battery|power|plug|android|version|off|factory|heat|start|phone place|install)"
regex = re.compile(patron)
phone = data_description.description.apply(lambda x: regex.search(x)) 
phoneClean = phone.apply(lambda x: 0 if x is None else 1) 
data_description["phone_issue"] = phoneClean 
data_description.phone_issue.value_counts()


# In[78]:


#histogram of 'phone_issue' variable
sns.histplot(data=data_description, x= 'phone_issue',stat='count', shrink=0.1,  discrete = True)


# In[79]:


#creation of new binary variable "extracamera_issue" (1 = issue, 0 = no issue)
patron = "(?P<extracamera>side|reporter|slave|endline|end lin|end-line|second phone|reporting)"
regex = re.compile(patron)
extracamera = data_description.description.apply(lambda x: regex.search(x)) 
extracameraClean = extracamera.apply(lambda x: 0 if x is None else 1) 
data_description["extracamera_issue"] = extracameraClean 
data_description.extracamera_issue.value_counts()


# In[80]:


#histogram of 'extracamera_issue' variable
sns.histplot(data=data_description, x= 'extracamera_issue',stat='count', shrink=0.1,  discrete = True)


# In[81]:


#creation of new binary variable "vod_issue" (1 = issue, 0 = no issue)
patron = "(?P<internalsystem>vod|troy|wowza|transcod|hermes|lisa|microsite|human|host app)"
regex = re.compile(patron)
vod = data_description.description.apply(lambda x: regex.search(x)) 
vodClean = vod.apply(lambda x: 0 if x is None else 1) 
data_description["vod_issue"] = vodClean 
data_description.vod_issue.value_counts()


# In[82]:


#histogram of 'vod_issue' variable
sns.histplot(data=data_description, x= 'vod_issue',stat='count', shrink=0.1,  discrete = True)


# In[83]:


#creation of new binary variable "poc_issue" (1 = issue, 0 = no issue)
patron = "(?P<poc>poc|unreachable|contact|not responsive|provider|no camera|kepit|tv production)"
regex = re.compile(patron)
poc = data_description.description.apply(lambda x: regex.search(x)) 
pocClean = poc.apply(lambda x: 0 if x is None else 1) 
data_description["poc_issue"] = pocClean 
data_description.poc_issue.value_counts()


# In[84]:


#histogram of 'poc_issue' variable
sns.histplot(data=data_description, x= 'poc_issue',stat='count', shrink=0.1,  discrete = True)


# In[85]:


#Deletion of unnecessary variables
data_description = data_description.drop(['description','issue_origin', 'issue_severity'], axis=1)
data_description


# In[86]:


data_description.columns


# In[87]:


#Variables converted into "int" format
data_description.astype(int)


# In[88]:


#New variables merged with original dataset
data = pd.concat([data, data_description], axis=1, join='inner')
data.head(3)


# In[89]:


#columns of final dataset
data.columns


# In[90]:


data.shape


# # New data obatined from external datasets

# ### Work with external dataset "numpartidos"
# 
# 'numpartidos.csv': contains information of other sport matches streamed at the same time that volleyball matches.
# 
# Variables are:
# - year: the year in which the match took place
# - month: the month in which the match took place
# - day_name: the weekday in which the match took place
# - interval_time: Match time duration 
# - production_method: Live streaming production
# - count: total number of matches streamed from different disciplines per date and time 

# In[91]:


import pandas as pd
import numpy as np
concurrency  = pd.read_csv('numpartidos.csv',encoding='unicode_escape')
concurrency


# In[92]:


#columns of dataset renamed
concurrency = concurrency.rename({'ï»¿YEAR':'year','MONTH':'months'
                                 ,'DAY_NAME':'weekday','INTERVAL_TIME':'2h_interval',
                                  'Count':'num_matches_interval' }, axis=1)
concurrency.columns


# In[93]:


#matches that belong to "Genius Live Production" streaming production
index =  concurrency.PRODUCTION_METHOD == 'Genius Live Production'

GeniusLivePconcurrency = concurrency[index]
GeniusLivePconcurrency.head()


# In[94]:


#column "PRODUCTION_METHOD" deleted
GeniusLivePconcurrency = GeniusLivePconcurrency.drop('PRODUCTION_METHOD', axis=1)


# In[95]:


#'num_matches_interval' column renamed as 'GLnum_matches_interval' (total number of matches streamed by Genius Live Production
#from different disciplines per date and time) 
GeniusLivePconcurrency = GeniusLivePconcurrency.rename({'num_matches_interval':'GLnum_matches_interval'},axis=1)
GeniusLivePconcurrency.head(3)


# In[96]:


#'GLnum_matches_interval' variable converted into "int" format
GeniusLivePconcurrency.astype({"GLnum_matches_interval": "int64" })


# In[97]:


# 'interval' column renamed as'2h_interval'
data = data.rename({'interval':'2h_interval'},axis=1)


# In[98]:


data.columns


# In[99]:


#Information of dataset "GeniusLivePconcurrency" merged with the original dataset
data =pd.merge(data,GeniusLivePconcurrency, on=['year', 'months', 'weekday', '2h_interval'], how='inner')
print(data.shape)
data.head(40)


# In[100]:


data.shape


# ### Creation of new variable "Total num_matches_interval"

# In[101]:


#Matches grouped per date and time at new variable "num_matches_interval"(total number of matches streamed 
#from different disciplines per date and time) 
concurrency = concurrency.groupby(['year', 'months', 'weekday', '2h_interval']).num_matches_interval.sum()
concurrency


# In[102]:


#reset of index
concurrency = concurrency.reset_index()
concurrency


# In[103]:


#"num_matches_interval" columns renamed as "Totalnum_matches_interval"
concurrency = concurrency.rename({'num_matches_interval':'Totalnum_matches_interval'},axis=1)


# In[104]:


#se añande la información nueva al dataset original
data =pd.merge(data,concurrency, on=['year', 'months', 'weekday', '2h_interval'], how='inner')
print(data.shape)
data.head(3)


# In[105]:


data.columns


# ### Work with external dataset "league_importance"
# 
# 'league_importance.csv': detailed league ranking
# 
# The variables are:
# - "league": match league
# - "classification": league ranking

# In[106]:


data_league_importance = pd.read_csv('league_importance.csv',encoding='unicode_escape', sep=',')
print(data_league_importance.dtypes)
data_league_importance.head(3)


# In[107]:


# 'clasificacion' column renamed as'clasification'
data_league_importance = data_league_importance.rename({'clasificacion':'classification'},axis=1)


# In[108]:


#League names list
data_league_importance.league.unique()


# In[109]:


#information from external dataset "data_important_league" merged with original dataset
data =pd.merge(data,data_league_importance, on='league', how='inner')
print(data.shape)
data.head()


# In[110]:


#Matches per classification
data.classification.value_counts()


# In[111]:


#"classification" values converted into numbers:
data = data.replace({'classification': {'A': 1, 'B': 2,'C':3,'D': 4}})
#"classification" column converted into "int" format
data = data.astype({"classification": "int64" }) 
data.head(3)


# ### Work with external dataset "data_operators"
# 
# 'data_operators.csv': contains number of stream operators per date
# 
# The variables are: 
# - Reference date: year and month in which the match took place
# - Attribute: day in which the match took place
# - Operator: Initials of designated operator per date
# - Value: 1= the assigned operator worked that day, 0= the assigned operator did not work that day

# In[112]:


import pandas as pd
import numpy as np
data_op  = pd.read_csv('data_operators.csv', encoding='unicode_escape')
data_op


# ### Creation of new variables: "num_operators","GL_num_match_day", "freq_rank_week_day_time", "freq_weekday_day_time" an "freq_month_day_time"

# In[113]:


#matches grouped per "Reference date" and 'Attribute' to obtain total number of operators per date at "Value" column
dataop = data_op.groupby(['ï»¿Reference date','Attribute']).Value.sum()
    
dataop = dataop.reset_index()


# In[114]:


dataop.head(3)


# In[115]:


#columns renamed
dataop = dataop.rename({'ï»¿Reference date': 'date', 'Attribute': 'day', 'Value':'num_operators'}, axis=1)
dataop.head(3)


# In[116]:


#Use of regular expressions to obtain year from "date" column
import re


pattern = "(?P<num>\d\d\d\d)"

year_regex = re.compile(pattern)


model_data_year = dataop.date.apply(str)


result_year = model_data_year.apply(lambda x: year_regex.search(x))

print(result_year.shape)


num_year = result_year.apply(lambda x: x if x is None else x.group())

#The information obtained was added to the new column: "year"
dataop["year"] = num_year
dataop


# In[117]:


# year 2020 added to rows without year information
dataop.year[dataop.year.isnull()] = 2020
dataop.year.isnull().sum()


# In[118]:


#Use of regular expressions to obtain month from "date" column
pattern = "(?P<month>\w*)"

month_regex = re.compile(pattern )


model_data_month = dataop.date.apply(str)


result_month = model_data_month.apply(lambda x: month_regex.search(x))

print(result_month.shape)


month = result_month.apply(lambda x: x if x is None else x.group())

#The information obtained was added to the new column: "month"
dataop["month"] = month
dataop.head()


# In[119]:


#no null values at "month" column
print(dataop.month.isnull().sum())
#values of "month" column
print(dataop.month.unique())


# In[120]:


#"month" values renamed with numbers:
dataop = dataop.replace({'April': '4', 'Aug':'8','August' : '8', 'December':'12', 'Feb':'2', 'February':'2', 'Jan':'1',
                'January':'1', 'Jul':'7', 'July':'7', 'Jun':'6', 'June':'6', 'March':'3', 'May':'5',
                'November':'11', 'October':'10', 'Sep':'9', 'September':'9' }) 


# In[121]:


#columns "year" and "month" converted into "int" format:
dataop = dataop.astype({"month": "int64", "year": "int64", }) 

dataop.month.unique()


# In[122]:


#"month" values renamed with names:
months   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Ago', 'Sep', 'Oct', 'Nov', 'Dec']

dataop['months'] = dataop.month.apply(lambda x: months[x-1])
dataop.head(4)


# In[123]:


#unnecessary columns dropped:
dataop = dataop.drop(['date','month'],axis=1)


# In[124]:


#new information merged with original dataset:
data =pd.merge(data,dataop, on=['year', 'months', 'day'], how='inner')
print(data.shape)
data.head(3)


# In[125]:


data.columns


# In[126]:


#matches grouped per weekday and time at new variable "freq_weekday_day_time":
frecuency_weekday = data.groupby(['weekday',"2h_interval"])
group_frecuency_weekday= pd.DataFrame(frecuency_weekday["date"].count())
print(group_frecuency_weekday)
data_merge_1=pd.merge(data,group_frecuency_weekday, on=['weekday',"2h_interval"])
data_merge_1.rename(columns={'date_y':'freq_weekday_day_time'}, inplace=True)
data_merge_1.head()


# In[127]:


#matches grouped per weekly rank and time at new variable "freq_rank_week_day_time":
frecuency_rank_week = data.groupby(['rank_week','2h_interval'])
group_rank_week= pd.DataFrame(frecuency_rank_week["date"].count())
print(group_rank_week)
data_merge_2=pd.merge(data_merge_1,group_rank_week, on=['rank_week','2h_interval'])
data_merge_2.rename(columns={'date':'freq_rank_week_day_time'}, inplace=True)
data_merge_2


# In[128]:


#matches grouped per month and time at new variable "freq_month_day_time":
frecuency_month_name = data.groupby(['months','2h_interval'])
group_month_name= pd.DataFrame(frecuency_month_name["date"].count())
print(group_month_name)
data_merge_3=pd.merge(data_merge_2,group_month_name, on=['months','2h_interval'])
data_merge_3.rename(columns={'date':'freq_month_day_time'}, inplace=True)
data_merge_3


# In[129]:


#dataframe renamed
data = data_merge_3


# 
# 

# In[130]:


#columns renamed
data = data.rename({'id':'match_id','time_x_x':'start_time', 'date_conv': 'date', 
                   'time_24_range': 'day_time', 'interval':'2h_interval', 'num_operators':'num_operators_day'}, axis=1)


# In[131]:


#descriptive analysis of final dataset
data.info()


# In[132]:


#Correlation heatmap between "issue" and the rest of variables:
plt.figure(figsize=(8,12))
sns.heatmap(data.corr()[['issue']], annot=True)


# In[133]:


data.columns


# In[134]:


#In order to do a simple analysis, "issue" was chosen as the Predictive Variable 
#unnecessary columns dropped:
data = data.drop(['GS_id',"date_x",'fixture_name','description', 'league_id', 'month','venue_name', 'issue_origin', 'issue_severity',
       'scoreboard_issue', 'internet_issues', 'schedule_issue', 
       'quality_issue', 'phone_issue', 'extracamera_issue', 'vod_issue',
       'poc_issue'], axis=1)

data.head(3)


# In[135]:


data.columns


# In[136]:


#columns organized
data = data[['match_id','venue_id','league',"classification",
             'country','continent','num_operators_day', 'date','start_time','year',
             'months','rank_week','day', 'weekday', 'day_time', 'freq_weekday_day_time',"freq_rank_week_day_time",'freq_month_day_time',
             '2h_interval', 'GLnum_matches_interval', 'Totalnum_matches_interval','issue']]


# In[137]:


data.columns


# In[138]:


data.to_csv('data_clean.csv')
data.info()


# ## Final columns: 
# 
# Predictive variables:
# - 'match_id': volleyball match id
# - 'venue_id': stadium id were the volleyball match was played
# - 'league': volleyball match league
# - "classification": league classification (1: higher classification, 4: lower classification).
# - "country": country were the volleyball match is played
# - "continent": continent were the volleyball match is played
# - 'num_operators_day':  total number of operators per date 
# - 'date': volleyball match date
# - 'start_time' : time of match start
# - 'year': the year in which the match took place
# - 'months': the month in which the match took place
# - 'rank_week': weekly number (variables are 1 (first week), 2(second week), 3(third week), and 4 (fourth week)).
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

# In[ ]:




