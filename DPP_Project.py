
# coding: utf-8

# In[6]:


import pandas as pd
import math
import numpy as np


# Import data from file

# In[12]:


data = pd.read_csv('adult.data', header=0, sep=', ', engine='python' )


# Drop these columns:
# - Fnlwgt
# - Education-num
# - Relationship
# - Capital gain
# - Capital loss
# - Hours-per-week

# In[13]:


data.drop(columns=['Fnlwgt', 'Education_num', 'Relationship', 'Capital_gain', 'Capital_loss', 'Hours_per_week'], inplace=True);


# Drop rows in which we have unknown values ('?')

# In[14]:


data.drop(data[
                (data.Workclass == '?') |
                (data.Education == '?') |
                (data.Marital_status == '?') |
                (data.Occupation == '?') |
                (data.Race == '?') |
                (data.Gender == '?') |
                (data.Native_country == '?') |
                (data.Salary == '?')].index, inplace=True);


# # Domain generalization
# 
# For each remaining Q.I. and its corresponding domain, we define a generalization hierarchy, according to this (numbers in parenthesis corresponds to the height of the associated generalization hierarchy)
# 
# - Age => 5-, 10-, 20- year ranges (4)
# - Workclass => Taxonomy tree (2)
# - Education =>  Taxonomy tree (2)
# - Marital status => Taxonomy tree (2)
# - Occupation => Taxonomy tree (2)
# - Race => Suppression (1)
# - Gender => Suppression (1)
# - Native country => Taxonomy tree (2)
# - Salary => Suppression (1)
# 
# Code's rule:
# 
# the less general version of the data of the Q.I. called 'pippo' is in data['pippo'] . 
# 
# For each generalization level n, we will have a variable called pippo_n containing the generalization of the values in data['pippo'], where pippo_1 is LESS general than pippo_2 

# ### Domain generalization for Age
# - 5- year ranges
# - 10- year ranges
# - 20- year ranges

# In[15]:


age_1 = (data['Age']/5).apply(math.floor)*5
age_2 = (data['Age']/10).apply(math.floor)*10
age_3 = (data['Age']/20).apply(math.floor)*20


# ### Domain generalization for Workclass
# 
# - Private => Private => Working
# - Self-emp-not-inc => Self-Emp => Working
# - Self-emp-inc => Self-Emp => Working
# - Federal-gov => Federal-gov => Working
# - Local-gov => Other-gov => Working
# - State-gov => Other-gov => Working
# - Without-pay => Not-Working => Not-Working
# - Never-worked => Not-Working => Not-Working

# In[16]:


workclass_1 = data['Workclass'].copy()
#Print unique values of this domain
#print(set(list(data['Workclass'])))

workclass_1.where(workclass_1 != 'Self-emp-inc', 'Self-Emp', inplace=True)
workclass_1.where(workclass_1 != 'Self-emp-not-inc', 'Self-Emp', inplace=True)
workclass_1.where(workclass_1 != 'Local-gov', 'Other-gov', inplace=True)
workclass_1.where(workclass_1 != 'State-gov', 'Other-gov', inplace=True)
workclass_1.where(workclass_1 != 'Never-worked', 'Not-Working', inplace=True)
workclass_1.where(workclass_1 != 'Without-pay', 'Not-Working', inplace=True)
#Print unique values of this domain
#print(set(list(workclass_1)))

workclass_2 = workclass_1.copy()
workclass_2.where(workclass_2 != 'Self-Emp', 'Working', inplace=True)
workclass_2.where(workclass_2 != 'Federal-gov', 'Working', inplace=True)
workclass_2.where(workclass_2 != 'Other-gov', 'Working', inplace=True)
workclass_2.where(workclass_2 != 'Private', 'Working', inplace=True)
#Print unique values of this domain
#print(set(list(workclass_2)))


# ### Domain generalization for Education
# 
# - "^10th" => "Dropout" => "Low"
# 
# - "^11th" => "Dropout" => "Low"
# 
# - "^12th" => "Dropout" => "Low"
# 
# - "^1st-4th" => "Dropout" => "Low"
# 
# - "^5th-6th" => "Dropout" => "Low"
# 
# - "^7th-8th" => "Dropout" => "Low"
# 
# - "^9th" => "Dropout" => "Low"
# 
# - "^Preschool" => "Dropout" => "Low"
# 
# - "^Assoc-acdm" => "Associates" => "High"
# 
# - "^Assoc-voc" => "Associates" => "High"
# 
# - "^Bachelors" => "Bachelors" => "High"
# 
# - "^Doctorate" => "Doctorate" => "High"
# 
# - "^HS-Grad" => "HS-Graduate" => "High"
# 
# - "^Masters" => "Masters" => "High"
# 
# - "^Prof-school" => "Prof-School" => "High"
# 
# - "^Some-college" => "HS-Graduate" => "High"
# 

# In[17]:


education_1 = data['Education'].copy()
#print(set(list(data['Education'])))

education_1.where(education_1 != '10th', 'Dropout', inplace=True)
education_1.where(education_1 != '11th', 'Dropout', inplace=True)
education_1.where(education_1 != '12th', 'Dropout', inplace=True)
education_1.where(education_1 != '1st-4th', 'Dropout', inplace=True)
education_1.where(education_1 != '5th-6th', 'Dropout', inplace=True)
education_1.where(education_1 != '7th-8th', 'Dropout', inplace=True)
education_1.where(education_1 != '9th', 'Dropout', inplace=True)
education_1.where(education_1 != 'Preschool', 'Dropout', inplace=True)
education_1.where(education_1 != 'Assoc-voc', 'Associates', inplace=True)
education_1.where(education_1 != 'Assoc-acdm', 'Associates', inplace=True)
education_1.where(education_1 != 'HS-grad', 'HS-Graduate', inplace=True)
education_1.where(education_1 != 'Some-college', 'HS-Graduate', inplace=True)
#print(set(list(education_1)))

education_2 = education_1.copy()
education_2.where(education_2 != 'Dropout', 'Low', inplace=True)
education_2.where(education_2 != 'Prof-school', 'High', inplace=True)
education_2.where(education_2 != 'Associates', 'High', inplace=True)
education_2.where(education_2 != 'Bachelors', 'High', inplace=True)
education_2.where(education_2 != 'Masters', 'High', inplace=True)
education_2.where(education_2 != 'HS-Graduate', 'High', inplace=True)
education_2.where(education_2 != 'Doctorate', 'High', inplace=True)
#print(set(list(education_2)))


# ### Domain generalization for Marital status
# 
# - Widowed => Widowed => Married
# - Divorced => Not-Married => Not-Married
# - Married-AF-spouse => Married => Married
# - Separated => Not-Married => Not-Married
# - Married-spouse-absent => Not-Married => Not-Married
# - Married-civ-spouse => Married => Married
# - Never-married => Never-Married => Not-Married

# In[18]:


marital_status_1 = data['Marital_status'].copy()
#print(set(list(data['Marital_status'])))

marital_status_1.where(marital_status_1 != 'Divorced', 'Not-Married', inplace=True)
marital_status_1.where(marital_status_1 != 'Married-AF-spouse', 'Married', inplace=True)
marital_status_1.where(marital_status_1 != 'Separated', 'Not-Married', inplace=True)
marital_status_1.where(marital_status_1 != 'Married-spouse-absent', 'Not-Married', inplace=True)
marital_status_1.where(marital_status_1 != 'Married-civ-spouse', 'Married', inplace=True)
#print(set(list(marital_status_1)))

marital_status_2 = marital_status_1.copy()
marital_status_2.where(marital_status_2 != 'Widowed', 'Married', inplace=True)
marital_status_2.where(marital_status_2 != 'Never-married', 'Not-Married', inplace=True)
#print(set(list(marital_status_2)))


# ### Domain generalization for Occupation
# - Adm-clerical => Admin => A
# - Armed-Forces => Military => A
# - Craft-repair => Blue-Collar => B
# - Exec-managerial => White-Collar => A
# - Farming-fishing => Blue-Collar => B
# - Handlers-cleaners => Blue-Collar => B
# - Machine-op-inspct => Blue-Collar => B
# - Other-service => Service => B
# - Priv-house-serv => Service => B
# - Prof-specialty => Other-Occupations => A
# - Protective-serv => Other-Occupations => A
# - Sales => Sales => B
# - Tech-support => Other-Occupations => A
# - Transport-moving => Other-Occupations => A

# In[19]:


occupation_1 = data['Occupation'].copy()
#print(set(list(data['Occupation'])))

occupation_1.where(occupation_1 != 'Adm-clerical', 'Admin', inplace=True)
occupation_1.where(occupation_1 != 'Armed-Forces', 'Military', inplace=True)
occupation_1.where(occupation_1 != 'Craft-repair', 'Blue-Collar', inplace=True)
occupation_1.where(occupation_1 != 'Exec-managerial', 'White-Collar', inplace=True)
occupation_1.where(occupation_1 != 'Farming-fishing', 'Blue-Collar', inplace=True)
occupation_1.where(occupation_1 != 'Handlers-cleaners', 'Blue-Collar', inplace=True)
occupation_1.where(occupation_1 != 'Machine-op-inspct', 'Blue-Collar', inplace=True)
occupation_1.where(occupation_1 != 'Other-service', 'Service', inplace=True)
occupation_1.where(occupation_1 != 'Priv-house-serv', 'Service', inplace=True)
occupation_1.where(occupation_1 != 'Prof-specialty', 'Other-Occupations', inplace=True)
occupation_1.where(occupation_1 != 'Protective-serv', 'Other-Occupations', inplace=True)
occupation_1.where(occupation_1 != 'Tech-support', 'Other-Occupations', inplace=True)
occupation_1.where(occupation_1 != 'Transport-moving', 'Other-Occupations', inplace=True)
#print(set(list(occupation_1)))

occupation_2 = occupation_1.copy()
occupation_2.where(occupation_2 != 'Military', 'A', inplace=True)
occupation_2.where(occupation_2 != 'Sales', 'B', inplace=True)
occupation_2.where(occupation_2 != 'Admin', 'A', inplace=True)
occupation_2.where(occupation_2 != 'White-Collar', 'A', inplace=True)
occupation_2.where(occupation_2 != 'Other-Occupations', 'A', inplace=True)
occupation_2.where(occupation_2 != 'Service', 'B', inplace=True)
occupation_2.where(occupation_2 != 'Blue-Collar', 'B', inplace=True)
#print(set(list(occupation_2)))


# ### Domain generalization for Race
# Based on suppression

# In[20]:


race_1 = data['Race'].copy()

race_1.where(race_1 != race_1 , '*', inplace=True)


# ### Domain generalization for Gender
# 
# Based on suppression

# In[51]:


gender_1 = data['Gender'].copy()

gender_1.where(gender_1 != gender_1, '*', inplace=True)


# ### Domain generalization for Native Country
# 
# - Cambodia => SE-Asia => Asia
# - Canada => British-Commonwealth => British-Commonwealth   
# - China => Asia => Asia
# - Columbia => South-America => South-America
# - Cuba => Latin-America => South-America
# - Dominican-Republic => Latin-America => South-America
# - Ecuador => South-America => South-America
# - El-Salvador => South-America => South-America
# - England => British-Commonwealth => British-Commonwealth
# - France => Euro_1 => Europe
# - Germany => Euro_1 => Europe
# - Greece => Euro_2 => Europe
# - Guatemala => Latin-America => South-America
# - Haiti => Latin-America => South-America
# - Holand-Netherlands => Euro_1 => Europe
# - Honduras => Latin-America => South-America
# - Hong => Asia => Asia
# - Hungary => Euro_2 => Europe
# - India => British-Commonwealth => British-Commonwealth
# - Iran => Asia => Asia
# - Ireland => British-Commonwealth => British-Commonwealth
# - Italy => Euro_1 => Europe
# - Jamaica => Latin-America => South-America
# - Japan => Asia => Asia
# - Laos => SE-Asia => Asia
# - Mexico => Latin-America => South-America
# - Nicaragua => Latin-America => South-America
# - Outlying-US(Guam-USVI-etc) => Latin-America => South-America
# - Peru => South-America => South-America
# - Philippines => SE-Asia => Asia
# - Poland => Euro_2 => Europe
# - Portugal => Euro_2 => Europe
# - Puerto-Rico => Latin-America => South-America
# - Scotland => British-Commonwealth => British-Commonwealth
# - South => Euro_2 => Europe
# - Taiwan => Asia => Asia
# - Thailand => SE-Asia => Asia
# - Trinadad&Tobago => Latin-America => South-America
# - United-States => United-States => United-States
# - Vietnam => SE-Asia => Asia
# - Yugoslavia => Euro_2 => Europe

# In[49]:


native_country_1 = data['Native_country'].copy()
#print(set(list(data['Native_country'])))

native_country_1.where(native_country_1 != 'Cambodia', 'SE-Asia', inplace=True)
native_country_1.where(native_country_1 != 'Canada', 'British-Commonwealth', inplace=True)
native_country_1.where(native_country_1 != 'China', 'Asia', inplace=True)
native_country_1.where(native_country_1 != 'Columbia', 'South-America', inplace=True)
native_country_1.where(native_country_1 != 'Cuba', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Dominican-Republic', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Ecuador', 'South-America', inplace=True)
native_country_1.where(native_country_1 != 'El-Salvador', 'South-America', inplace=True)
native_country_1.where(native_country_1 != 'England', 'British-Commonwealth', inplace=True)
native_country_1.where(native_country_1 != 'France', 'Euro_1', inplace=True)
native_country_1.where(native_country_1 != 'Germany', 'Euro_1', inplace=True)
native_country_1.where(native_country_1 != 'Greece', 'Euro_2', inplace=True)
native_country_1.where(native_country_1 != 'Guatemala', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Haiti', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Holand-Netherlands', 'Euro_1', inplace=True)
native_country_1.where(native_country_1 != 'Honduras', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Hong', 'Asia', inplace=True)
native_country_1.where(native_country_1 != 'Hungary', 'Euro_2', inplace=True)
native_country_1.where(native_country_1 != 'India', 'British-Commonwealth', inplace=True)
native_country_1.where(native_country_1 != 'Iran', 'Asia', inplace=True)
native_country_1.where(native_country_1 != 'Ireland', 'British-Commonwealth', inplace=True)
native_country_1.where(native_country_1 != 'Italy', 'Euro_1', inplace=True)
native_country_1.where(native_country_1 != 'Jamaica', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Japan', 'Asia', inplace=True)
native_country_1.where(native_country_1 != 'Laos', 'SE-Asia', inplace=True)
native_country_1.where(native_country_1 != 'Mexico', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Nicaragua', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Outlying-US(Guam-USVI-etc)', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Peru', 'South-America', inplace=True)
native_country_1.where(native_country_1 != 'Philippines', 'SE-Asia', inplace=True)
native_country_1.where(native_country_1 != 'Poland', 'Euro_2', inplace=True)
native_country_1.where(native_country_1 != 'Portugal', 'Euro_2', inplace=True)
native_country_1.where(native_country_1 != 'Puerto-Rico', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Scotland', 'British-Commonwealth', inplace=True)
native_country_1.where(native_country_1 != 'South', 'Euro_2', inplace=True)
native_country_1.where(native_country_1 != 'Taiwan', 'Asia', inplace=True)
native_country_1.where(native_country_1 != 'Thailand', 'SE-Asia', inplace=True)
native_country_1.where(native_country_1 != 'Trinadad&Tobago', 'Latin-America', inplace=True)
native_country_1.where(native_country_1 != 'Vietnam', 'SE-Asia', inplace=True)
native_country_1.where(native_country_1 != 'Yugoslavia', 'Euro_2', inplace=True)
#print(sorted(set(list(native_country_1))))

native_country_2 = native_country_1.copy()
native_country_2.where(native_country_2 != 'Euro_1', 'Europe', inplace=True)
native_country_2.where(native_country_2 != 'Euro_2', 'Europe', inplace=True)
native_country_2.where(native_country_2 != 'SE-Asia', 'Asia', inplace=True)
native_country_2.where(native_country_2 != 'Latin-America', 'South-America', inplace=True)
#print(sorted(set(list(native_country_2))))


# ### Domain generalization for Salary
# 
# Based on suppression

# In[55]:


salary_1 = data['Salary'].copy()

salary_1.where(salary_1 != salary_1, '*', inplace=True)

