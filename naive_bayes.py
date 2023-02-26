# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:05:14 2023

@author: stanley
"""

# import all of the used packadges

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import norm
import streamlit as st
import plotly_express as px
from streamlit_option_menu import option_menu
st.set_page_config(layout="wide")

# creating option menu for the webapp

with st.sidebar: 
	sideselect = option_menu(
		menu_title = 'Navigation Pane',
		options = ["Start", "Intro", "Data Selection", "Explaining Naive Bayes", 'Interactive Classifier', 'Data Analysis', 'Conclusion'],
		menu_icon = 'arrow-down-right-circle-fill',
		icons = ['check-lg', 'book', 'box', 'map', 'boxes', 'bar-chart', 
		'check2-circle'],
		default_index = 0,
		)

# load in dataset + basic cleaning

untouched = pd.read_csv('D:\heart.csv')
df = pd.read_csv('D:\heart.csv').astype(int)
df = df.drop('Diabetes', axis = 1)
num_cols = ['BMI', 'Age']

# defining categorical dataset

df_cat = df.drop(['HeartDiseaseorAttack', 'BMI', 'Age'], axis = 1)

# create column dictionary

pretty_names = ['High Blood Pressure', 'High Cholesterol', 'Cholesterol Check', 'Smoker', 'Stroke', 'Physical Activity', 'Fruits', 'Veggies', 'Heavy Alcohol Consumption', 'Any Healthcare', 'Poor', 'General Health', 'Mental Health', 'Physical Health', 'Difficulty in Walking', 'Gender', 'Education', 'Income']
cat_col_dict = {col:pretty_names[i] for i, col in enumerate(df_cat.columns)}

num_col_dict = {'BMI':'BMI', 'Age':'Age'}

all_col_dict = cat_col_dict.copy()
all_col_dict.update(num_col_dict)

# extracting the target list

target_list = df['HeartDiseaseorAttack'].astype(int)

# create chi square dictionary

dict_ = {col:chi2_contingency(pd.crosstab(df['HeartDiseaseorAttack'], df[col]))[0:2] for col in df_cat.columns}

# creating test set

test_set = df.drop(['CholCheck', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'MentHlth'], axis = 1)

# writing GUIs for each selection

if sideselect == 'Start':
    
    
    st.title('Starting Page')
    st.write("To start, please open the sidebar and select a section.")


if sideselect == 'Intro':
    
    
    st.title('Introduction')
    st.header('What is Heart Disease?')
    st.markdown("There are all kinds of heart diseases, but in daily life, if we talk about heart disease, we are usually referring to [coronary artery disease (CDC official page)](https://www.cdc.gov/heartdisease/coronary_ad.htm).")
    st.write("This disorder can influence the blood flow to the heart, normally decreased flow, which may cause heart attacks.")
    
    
    st.header('Why is it so important?')
    st.write("Heart attacks are killers of mankind. In America alone, heart attacks were the cause of **every 1 out of 4 deaths** in 2020.")
    st.write("Some symptoms of heart diseases include chest pain or discomfort, upper back or neck pain, indigestion, heartburn, nausea or vomiting, extreme fatigue, upper body discomfort, dizziness and shortness of breath. *Source: Centers for Disease Control and Prevention*")
    st.info('*Did you know that every 40 seconds, someone in the United States has a heart attack?*')
    
    
    st.header("What is this app about, then?")
    st.write("In this app, I will try to produce a machine learning model that uses the Bayes Theorm of conditional probability to predict heart diseases given the features that you input using the easy interface that `streamlit` provides.")
    st.write("I will also tell you about how I made this model. If you want to use the predictor immediately, you can open the sidebar and select the Interactive Classifier section.")
    
if sideselect == 'Data Selection':
    
    
    st.title('Data Selection')
    st.write("Data is the determining factor for a model's accuracy. Bad data often leads to bad accuracy.")
    st.write("So, selecting the right source of data and the right piece of data to train is a process which does not allow any carelesness.")
    st.write("For this project, I chose a dataset from the prestigious data science website Kaggle. Kaggle is a friendly platform that has huge amounts of datasets that you can download right away for free. The link to the dataset is below:")
    st.markdown("[Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)")
    
    
    st.header("Data Exploration")
    st.write("This is the whole untouched dataset:")
    st.write(pd.read_csv('D:\heart.csv'))
    st.info("*You can adjust the display of the dataset's size by dragging the small white square in the bottom right corner of the dataset. You can also adjust the columns' width.*")
    st.write("Ok, let's dive right into this dataset.")
    st.write("This dataset consists of 22 columns with one columns indicating whether the patient has heart disease or not. So, we are left with 21 features that we can use for prediction.")
    st.write("We must be careful when it comes to selecting features for prediction. Feature selection is a very important and inevitable part of building a machine learning model. If we select the right features which usually have the most relation with the prediction column, we can achieve high accuracies easily. But if we aren't careful, it may lead to bad consequences.")
    st.write("We will do a chi-squared test on all of the columns to select the 5 most related features. ")
    st.info("Chi-squared tests are experiments that can show us how strong a relationship a feature has with its prediction column.")
    
    
    st.header("Column Explaination")
    st.write("Not every dataset on Kaggle is perfect! Like this one. In this dataset, there are lots of column names that we do not know the meaning of, for example: NoDocbcCost or CholCheck")
    st.write("That's why I will make an easy-going table to tell you what every column is about:")
    st.write({'HeartDiseaseorAttack':'Does the patient have heart disease or attack? (target column)', 'HighBP':'Does the patient have high blood pressure?', 'HighChol':'Does the patient have high cholesterol?', 'CholCheck':'Recent cholesterol check?', 'BMI':'The universal ratio between weight and height', 'Smoker':'Does the patient smoke?', 'Stroke':'Does the patient have a stroke history?', 'Diabetes':'Does the patient have diabetes?', 'PhysActivity':'Do you have daily physical activity?', 'Fruits':'Does the patient eat fruits on a daily basis?', 'Veggies':'Does the patient consume vegetables on a daily basis?', 'HvyAlcoholConsump':'Does the patient drink heavily?', 'AnyHealthcare':'Does the patient have any health care insurances?', 'NoDocbcCost':"Is there any occasion when the patient didn't go to see a doctor because they can't afford it?", 'GenHlth':'General health rating 1-5', 'MentHlth':'Mental health rating 0-30', 'PhysHlth':'Physical health rating 0-30', 'DiffWalk':'Does the patient have a difficulty in walking?', 'Sex':"What is the patient's sex? 0 for female, 1 for male", 'Age':'1 for age 18 to 24, 2 for age 25 to 29, 3 for age 30 to 34, 4 for age 35 to 39 ... 12 for age 75 to 79, 13 for age 80 or older', 'Education':'Rate your education 0-6', 'Income':'Rate income 0-8'})
    

if sideselect == 'Explaining Naive Bayes':
    
    
    st.title('What is Naive Bayes?')
    st.header('Introduction to Naive Bayes')
    st.write("Naive Bayes classifiers are machine learning models that can predict things without actual data training. Naive Bayes classifiers are a series or a family of multiple models that all use the **Bayes Theorem** as a base rule. This rule requires that every pair of features being classified is independent of each other.")
    
    
    st.header('The *ULTIMATE* Bayes Theorem')
    st.write("The Bayes Theorem is as follows:")
    st.latex(r'''
             P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
    ''')
    st.write("The Bayes theorem is a mathematical formula that helps us update our beliefs about something when we learn new information. It's often used in probability and statistics to help us make decisions based on the available data.")
    st.success("This theorem means **the probability of A event happening given that B event has already happened is equal to the probability of B event happening given A event has already happened times the probability of A event happening on its own divided by the probability of B event happening on its own.**")
    
    
    st.header("Pros and Cons of Naive Bayes Classifiers")
    st.subheader("Pros first")
    st.write("1. **Simplicity**: Naive Bayes Classifiers are very easy to understand and therefore simple to create from scratch, which makes it very popular with beginners.")
    st.write("2. **Effeciency**: Naive Bayes Classifiers are very efficient, they don't even need training.")
    st.write("3. **Robustness**: Naive Bayes can handle a lot of noise and missing data, while other classifiers may not be so adaptive.")
    st.write("4. **Scalability**: Naive Bayes can scale well with lots of data or minimal data while other classifiers may become impractical as the number of features or classes grows.")
    st.subheader("Then Cons")
    st.write("1. **Assumption of Independence**: Naive Bayes classifiers are based on the idea that all features are independent of each other.")
    st.write("2. **Sensitivity**: Even if it is good with coping with bad data, Naive Bayes classifiers are sensitive to outliers and imbalanced data.")
    st.write("3. **Biased Class Probabilities**: Naive Bayes could become biased on prediction if you give it biased data.")
    
    
    
if sideselect == 'Interactive Classifier':
    st.title('Interactive Classifier')
    options = st.multiselect(
    "Please select some features to predict heart disease: (if you don't select anything, the model will go with default)",
    all_col_dict.values(),
    ['General Health', 'Difficulty in Walking', 'High Blood Pressure', 'Stroke'])
    st.write('Your choices:', options)
    
    # the NAIVE BAYES FUNCTION
    def naive_bayes(df):
        target_col_name = 'HeartDiseaseorAttack'
        num_col_names = [col for col in df.columns if col in num_col_dict.keys()]
        
        
        def fit_distribution(data):
            mu = np.mean(data)
            sigma = np.std(data)
            dist = norm(mu, sigma)
            return dist
        
        
        df = df.dropna()
        X = df.drop([target_col_name], axis = 1)
        numcols = num_col_names
        catcols = [i for i in X.columns if i not in numcols]
        df[target_col_name] = df[target_col_name].astype(int)
        X0 = df[df[target_col_name] == 0]
        X1 = df[df[target_col_name] == 1]
        X0 = X0.drop([target_col_name], axis = 1)
        X1 = X1.drop([target_col_name], axis = 1)
        
        
        chisquare_dict = {col:chi2_contingency(pd.crosstab(df[target_col_name], df[col]))[0:2] for col in df[catcols].columns}
        sort_dict = {column:chisquare_dict[column][0] for column in df[catcols].columns}
        sorted_thing = sorted(sort_dict.items(), reverse = True, key = lambda e:e[1])
        resulting_cols = [sorted_thing[0:4][i][0] for i in np.arange(0, 4, 1)]
        
        
        for col in resulting_cols:
            X0[col] = X0[col].replace(dict(X0[col].value_counts(normalize = True)))
        for col in resulting_cols:
            X1[col] = X1[col].replace(dict(X1[col].value_counts(normalize = True)))
        
        
        numcol_dict0 = {col:fit_distribution(X0[col]).pdf(X[col]) for col in numcols}
        catcol_dict0 = {col:np.array(X[col]) for col in resulting_cols}
        numcol_df0 = pd.DataFrame(numcol_dict0)
        catcol_df0 = pd.DataFrame(catcol_dict0)
        pred_df0 = pd.concat([catcol_df0, numcol_df0], axis = 1)
        pred_series0 = pred_df0.prod(axis = 1)
        
        
        numcol_dict1 = {col:fit_distribution(X1[col]).pdf(X[col]) for col in numcols}
        catcol_dict1 = {col:np.array(X[col]) for col in resulting_cols}
        numcol_df1 = pd.DataFrame(numcol_dict1)
        catcol_df1 = pd.DataFrame(catcol_dict1)
        pred_df1 = pd.concat([catcol_df1, numcol_df1], axis = 1)
        pred_series1 = pred_df1.prod(axis = 1)
        
        
        preds = (pred_series0 < pred_series1) * 1
        accuracy = np.mean(preds == df[target_col_name])
        st.write('The accuracy is:', accuracy)  
        
        
        return pd.DataFrame(zip(df[target_col_name], preds), columns = ['Actual', 'Preds'])
    
    with st.form('Submit features to predict heart attack:'):
        options_df = [k for k, v in all_col_dict.items() if v in options]
        submitted = st.form_submit_button('Submit features')
        if submitted:
            st.dataframe(naive_bayes(df[options_df + ['HeartDiseaseorAttack']]))