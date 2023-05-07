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


# load in dataset + basic cleaning

untouched = pd.read_csv('heart.csv')
df = pd.read_csv('heart.csv').astype(int)
df = df.drop('Diabetes', axis = 1)
num_cols = ['BMI', 'Age',  'GenHlth', 'MentHlth', 'PhysHlth', 'Education', 'Income']


# defining categorical dataset

df_cat = df.drop(num_cols, axis = 1)
df_cat = df_cat.drop(['HeartDiseaseorAttack'], axis = 1)
df.loc[:, df_cat.columns] = df.loc[:, list(df_cat.columns)].replace({1: 'Yes', 0: 'No'})
df_cat_sun = df.copy()
df_cat_sun = df.drop(num_cols, axis = 1)
df_cat_sun['HeartDiseaseorAttack'].replace({1: 'Yes', 0: 'No'}, inplace = True)


# create column dictionary

pretty_names = ['High Blood Pressure', 'High Cholesterol', 'Cholesterol Check', 'Smoker', 'Stroke', 'Physical Activity', 'Fruits', 'Veggies', 'Heavy Alcohol Consumption', 'Any Healthcare', 'Poor', 'Difficulty in Walking', 'Gender']
cat_col_dict = {col:pretty_names[i] for i, col in enumerate(df_cat.columns)}

target_cat_dict = {'HeartDiseaseorAttack':'Heart Disease'}
target_cat_dict.update(cat_col_dict)
for name, pname in target_cat_dict.items():
    df_cat_sun[name] = pname + ' ' + df_cat_sun[name]

num_col_dict = {'BMI':'BMI', 'Age':'Age', 'GenHlth':'General Health', 'MentHlth':'Mental Health', "PhysHlth":"Physical Health", 'Education':'Education', 'Income':'Income'}

all_col_dict = cat_col_dict.copy()
all_col_dict.update(num_col_dict)
target_col_dict = all_col_dict.copy()
target_col_dict.update({'HeartDiseaseorAttack':'Heart Disease'})


# extracting the target list

target_list = df['HeartDiseaseorAttack'].astype(int)

# create chi square dictionary

dict_ = {col:chi2_contingency(pd.crosstab(df['HeartDiseaseorAttack'], df[col]))[0:2] for col in df_cat.columns}


# creating option menu for the webapp

with st.sidebar: 
	sideselect = option_menu(
		menu_title = 'Navigation Pane',
		options = ["Start", "Intro", "Data Cleaning", "Data Exploration", "Explaining Naive Bayes", 'Interactive Classifier', 'Feature Selection', 'Conclusion'],
		menu_icon = 'arrow-down-right-circle-fill',
		icons = ['check-lg', 'book', 'clipboard', 'compass', 'map', 'boxes', 'cart', 'check2-circle'],
		default_index = 0,
		)


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
    st.write("Heart attacks are killers of mankind. In America alone, heart attacks were the cause of **every 1 out of 4 deaths** in 2020 (for COVID? 1 out of every 2.5 deaths).")
    st.write("Some symptoms of heart diseases include chest pain or discomfort, upper back or neck pain, indigestion, heartburn, nausea or vomiting, extreme fatigue, upper body discomfort, dizziness and shortness of breath. *Source: Centers for Disease Control and Prevention*")
    st.info('*Did you know that every 40 seconds, someone in the United States has a heart attack?*')
    
    
    st.header("What is this app about, then?")
    st.write("In this app, I will try to produce a machine learning model that uses the Bayes Theorm of conditional probability to predict heart diseases given the features that you input using the easy interface that `streamlit` provides.")
    st.write("I will also tell you about how I made this model. If you want to use the predictor immediately, you can open the sidebar and select the Interactive Classifier section.")
    
    
    st.header("Choosing a dataset")
    st.write("For this project, I chose a dataset from the prestigious data science website Kaggle. Kaggle is a friendly platform that has huge amounts of datasets that you can download right away for free. The link to the dataset is below:")
    st.markdown("[Heart Disease Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset)")
    
    
if sideselect == "Data Cleaning":
    
    
    st.title("Data Cleaning")
    st.write("Data cleaning is a process all data scientists should do before training a model. This process includes the removing of biased data and null data, normalizing the numerical columns and making all of the units the same.")
    st.write("In this section, I will be talking about the cleaning process of my data. This is pretty important because with out data cleaning, a model won't be accurate at all (depending on what type of model it is)")
    
    
    st.header("Converting Column Names for Classifier")
    st.write("In this part I will talk about how I converted the column names of my dataset to something that you guys can read as a user of the interactive classifier.")
    st.write("First, let's take a look at the column names of this dataset.")
    st.write(untouched.columns)
    st.write("As you can see there are lots of column names that are hard to understand. For example, Nodocbccost or Diffwalk. Also, I can't keep guessing at the column names' meaning because that may lead to misunderstanding of the column names which then will cause unimaginable problems for the classifier.")
    st.write("So, I dove back into Kaggle, searching for the meaning of these columns.")
    st.write("And boom! I found that there are also [other people](https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset/discussion/284985) who are confused about the column names.")
    st.write("There's a [solution](https://www.kaggle.com/code/alexteboul/heart-disease-health-indicators-dataset-notebook) to this annoying problem though. The author of the dataset posted a notebook in the code section of this dataset. The author shows us how he got the original data from a website and did some intense cleaning on it.")
    st.write("That's where I got some idea for the column names' meaning.")
    st.write("For a Interactive Classifier that lets users predict their own heart status, I created two dictionaries that points normal column names to their original confusing column names in the dataset.")
    st.code('''
            pretty_names = ['High Blood Pressure', 'High Cholesterol', 'Cholesterol Check', 'Smoker', 'Stroke', 'Physical Activity', 'Fruits', 'Veggies', 'Heavy Alcohol Consumption', 'Any Healthcare', 'Poor', 'General Health', 'Mental Health', 'Physical Health', 'Difficulty in Walking', 'Gender', 'Education', 'Income']
cat_col_dict = {col:pretty_names[i] for i, col in enumerate(df_cat.columns)}
    ''')
    st.write("These are column names for categorical columns and here are the numeric columns' names:")
    st.code("num_col_dict = {'BMI':'BMI', 'Age':'Age'}")
    st.write("Now, you can take a look at the column names in the Interactive Classifier part. They are very pretty indeed!")
    
    
    st.write("This is the whole untouched dataset:")
    st.write(pd.read_csv('D:\heart.csv'))
    st.info("*You can adjust the display of the dataset's size by dragging the small white square in the bottom right corner of the dataset. You can also adjust the columns' width.*")
    
    
    st.header("Column Explaination")
    st.write("Not every dataset on Kaggle is perfect! For example, this one. In this dataset, there are lots of column names that we do not know the meaning of, for example: NoDocbcCost or CholCheck")
    st.write("That's why I will make an easy-going table to tell you what every column is about:")
    st.write({'HeartDiseaseorAttack':'Does the patient have heart disease or attack?', 'HighBP':'Does the patient have high blood pressure?', 'HighChol':'Does the patient have high cholesterol?', 'CholCheck':'Recent cholesterol check?', 'BMI':'The universal ratio between weight and height', 'Smoker':'Does the patient smoke?', 'Stroke':'Does the patient have a stroke history?', 'Diabetes':'Does the patient have diabetes?', 'PhysActivity':'Do you do daily physical activity?', 'Fruits':'Does the patient eat fruits on a daily basis?', 'Veggies':'Does the patient consume vegetables on a daily basis?', 'HvyAlcoholConsump':'Does the patient drink heavily?', 'AnyHealthcare':'Does the patient have any health care insurances?', 'NoDocbcCost':"Is there any occasion when the patient didn't go see a doctor because they can't afford it?", 'GenHlth':'General health rating 1-5', 'MentHlth':'Mental health rating 0-30', 'PhysHlth':'Physical health rating 0-30', 'DiffWalk':'Does the patient have a difficulty in walking?', 'Sex':"What is the patient's sex? 0 for female, 1 for male", 'Age':'1 for age 18 to 24, 2 for age 25 to 29, 3 for age 30 to 34, 4 for age 35 to 39 ... 12 for age 75 to 79, 13 for age 80 or older', 'Education':'Rate your education 0-6', 'Income':'Rate income 0-8'})
    
    
if sideselect == 'Data Exploration':
    
    
    st.title('Data Exploration')
    col1, col2 = st.columns([2,6])
    col1.header("Custom Categorical Histogram")
    col1.write("Experiment with the following dropdowns to produce a custom histogram.")
    container2 = col2.empty()
    with st.form("Submit", clear_on_submit = True):
        xval_p = col1.selectbox("Please select a category column for the x-axis:", all_col_dict.values(), index = 16)
        colorval_p = col1.selectbox("Please select a different category column for the color group:", cat_col_dict.values())
        percentval = col1.checkbox("Check if you want percents and not counts")
        xval = [k for k,v in all_col_dict.items() if v == xval_p][0]
        colorval = [k for k,v in cat_col_dict.items() if v == colorval_p][0]
        hist_default = px.histogram(df, x = 'Age', title = 'Interactive Histogram')
        hist_default.update_traces(marker_line_width = 2)
        container2.plotly_chart(hist_default, use_container_width = True)
        sub = st.form_submit_button('Submit To Create Histogram')
        if sub:
            if percentval:
                fig1 = px.histogram(df, x = xval, color = colorval, labels = all_col_dict, barmode = 'group', barnorm = 'percent', title = 'Interactive Histogram')
                container2.plotly_chart(fig1, use_container_width = True)
            else:
                fig1 = px.histogram(df, x = xval, color = colorval, labels = all_col_dict, barmode = 'group', title = 'Interactive Histogram', hover_data = target_col_dict.keys())
                container2.plotly_chart(fig1, use_container_width = True)
    
    
    col5, col6 = st.columns([3, 5])
    container6 = col6.empty()
    col5.header('Interactive Sunburst Chart')
    col5.write('A Sunburst chart is a multiple layered pie chart, it is made up of rings which can show hierarchy or flow.')
    col5.write("Sometimes, pie charts or bar charts couldn't handle your complex data. That's when the Sunburst chart comes into place.")
    container6.plotly_chart(px.sunburst(df_cat_sun, path = ['HeartDiseaseorAttack', 'HighBP', 'HighChol'], labels=all_col_dict, title = 'Interactive Sunburst Plot'), use_container_width = True)
    with st.form("interactive sun"):
        sunlist = col5.multiselect('Please select 2 columns for the sunburst chart:', cat_col_dict.values(), max_selections=2)
        subsun = st.form_submit_button("Submit to Create Sunburst Chart")
        if subsun:
            selected_sun = [k for k,v in cat_col_dict.items() if v in sunlist]
            fig = px.sunburst(df_cat_sun, path=['HeartDiseaseorAttack'] + selected_sun, labels=target_col_dict, title = 'Interactive Sunburst Plot')
            fig.update_traces(texttemplate='%{labels}<br>%{count}', textposition='middle center')
            container6.plotly_chart(fig, use_container_width = True)
            
    
    col7, col8 = st.columns([3, 5])
    container8 = col8.empty()
    col7.header("Interactive Box Plot")
    col7.write("A box plot (or box and whisker plot) uses boxes and lines to describe the distributions of one or lots of groups of numerical data, like BMI or Age. Box limits shows the range of the central 50% of the data, with a line in the box marking the median value. Lines extend from each box to capture the range of the remaining data, with dots placed past the line edges to indicate outliers.")
    col7.write('')
    container8.plotly_chart(px.box(df, x = 'HeartDiseaseorAttack', y = 'Age', color = 'HeartDiseaseorAttack', title = 'Interactive Box Plot'), use_container_width = True)
    with st.form('ok box'):
        boxlist = col7.multiselect('Please select a column for the y-axis of the box plot:', num_col_dict.values(), max_selections = 1)
        subbox = st.form_submit_button("Submit to Create Box Plot")
        if subbox:
            selected_box = [k for k,v in num_col_dict.items() if v in boxlist]
            container8.plotly_chart(px.box(df, x = 'HeartDiseaseorAttack', y = selected_box[0], color = 'HeartDiseaseorAttack', title = 'Interactive Box Plot'), use_container_width = True)
                        

if sideselect == 'Explaining Naive Bayes':
    
    
    st.title('What is Naive Bayes?')
    st.header('Introduction to Naive Bayes')
    st.write("Naive Bayes classifiers are machine learning models that can predict things without actual data training. Naive Bayes classifiers are a series or a family of multiple models that all use the **Bayes Theorem** as a base rule. This rule requires that every pair of features being classified is independent of each other.")
    
    
    st.header('The *ULTIMATE* Bayes Theorem')
    st.write("The Bayes Theorem is as follows:")
    st.latex(r'''
             P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
    ''')
    st.write("The Bayes theorem is a mathematical formula that helps us update our beliefs about something when we learn new information about our data. It's often used in probability and statistics to help us make decisions based on the available data.")
    st.success("This theorem means **the probability of A event happening given that B event has already happened is equal to the probability of B event happening given A event has already happened times the probability of A event happening on its own divided by the probability of B event happening on its own.**")
    
    
    st.header("Pros and Cons of Naive Bayes Classifiers")
    st.subheader("Pros first")
    st.write("1. **Simplicity**: Naive Bayes Classifiers are very easy to understand and therefore simple to create from scratch, which makes it very popular with beginners (like me).")
    st.write("2. **Effeciency**: Naive Bayes Classifiers are very efficient, they don't even need training, just a dataset, that's all they need.")
    st.write("3. **Robustness**: Naive Bayes can handle a lot of noise and missing data, while other classifiers may not be so adaptive.")
    st.write("4. **Scalability**: Naive Bayes can scale well with lots of data or minimal data while other classifiers may become impractical as the number of features or classes grows.")
    st.subheader("Then Cons")
    st.write("1. **Assumption of Independence**: Naive Bayes classifiers are based on the idea that all features are independent of each other.")
    st.write("2. **Sensitivity**: Even if it is good with coping with bad data, Naive Bayes classifiers are sensitive to outliers and imbalanced data.")
    st.write("3. **Biased Class Probabilities**: Naive Bayes could become biased on prediction if you give it biased data.")
    
    
    
if sideselect == 'Interactive Classifier':
    st.title('Interactive Classifier')
    options = st.multiselect(
    "Please select some features to predict heart disease: (if you don't select anything, the model will go with default preferences, which is General Health, Difficulty in Walking, High Blood Pressure and Stroke)",
    all_col_dict.values(),
    ['General Health', 'Difficulty in Walking', 'High Blood Pressure', 'Stroke'])
    st.write('Your choices:', options)
    
    # the NAIVE BAYES FUNCTION
    def naive_bayes(df):
        target_col_name = 'HeartDiseaseorAttack'
        num_col_names = [col for col in df.columns if col in num_col_dict.keys()]
        
        
        # the fit distribution function used on numerical data
        def fit_distribution(data):
            mu = np.mean(data)
            sigma = np.std(data)
            dist = norm(mu, sigma)
            return dist
        
        
        df = df.dropna().copy()
        X = df.drop([target_col_name], axis = 1)
        numcols = num_col_names
        catcols = [i for i in X.columns if i not in numcols]
        df[target_col_name] = df[target_col_name].astype(int)
        X0 = df[df[target_col_name] == 0]
        X1 = df[df[target_col_name] == 1]
        X0 = X0.drop([target_col_name], axis = 1)
        X1 = X1.drop([target_col_name], axis = 1)
        
        
# =============================================================================
#         chisquare_dict = {col:chi2_contingency(pd.crosstab(df[target_col_name], df[col]))[0:2] for col in df[catcols].columns}
#         sort_dict = {column:chisquare_dict[column][0] for column in df[catcols].columns}
#         sorted_thing = sorted(sort_dict.items(), reverse = True, key = lambda e:e[1])
#         resulting_cols = [sorted_thing[0:4][i][0] for i in np.arange(0, 4, 1)]
# =============================================================================
        
        
        for col in catcols:
            X0[col] = X0[col].replace(dict(X0[col].value_counts(normalize = True)))
        for col in catcols:
            X1[col] = X1[col].replace(dict(X1[col].value_counts(normalize = True)))
        
        
        numcol_dict0 = {col:fit_distribution(X0[col]).pdf(X[col]) for col in numcols}
        catcol_dict0 = {col:np.array(X[col]) for col in catcols}
        numcol_df0 = pd.DataFrame(numcol_dict0)
        catcol_df0 = pd.DataFrame(catcol_dict0)
        pred_df0 = pd.concat([catcol_df0, numcol_df0], axis = 1)
        pred_series0 = pred_df0.prod(axis = 1)
        
        
        numcol_dict1 = {col:fit_distribution(X1[col]).pdf(X[col]) for col in numcols}
        catcol_dict1 = {col:np.array(X[col]) for col in catcols}
        numcol_df1 = pd.DataFrame(numcol_dict1)
        catcol_df1 = pd.DataFrame(catcol_dict1)
        pred_df1 = pd.concat([catcol_df1, numcol_df1], axis = 1)
        pred_series1 = pred_df1.prod(axis = 1)
        
        
        preds = (pred_series0 < pred_series1) * 1
        accuracy = np.mean(preds == df[target_col_name])
        st.write('The accuracy is:', accuracy)
        st.write('The number of zeros predicted is', preds.value_counts())
        
        
        return pd.DataFrame(zip(df[target_col_name], preds), columns = ['Actual', 'Preds'])
    
    with st.form('Submit features to predict heart attack:'):
        options_df = [k for k, v in all_col_dict.items() if v in options]
        submitted = st.form_submit_button('Submit features')
        if submitted:
            st.dataframe(naive_bayes(df[options_df + ['HeartDiseaseorAttack']]))
            st.write("This is the accuracy for the Naive Bayes classifier that you just created with the features that you have selected.")
        
        
if sideselect == 'Feature Selection':
    st.title('Feature Selection')
    st.write("Data is the determining factor for a model's accuracy. Bad data often leads to bad accuracy.")
    st.write("So, selecting the right source of data and the right piece of data to train is a process which does not allow any carelesness.")
    st.write("Ok, let's dive right into this dataset.")
    st.write("This dataset consists of 22 columns. The first column is the column for prediction. So, we are left with 21 features that we can use for prediction.")
    st.write("We must be careful when it comes to selecting features for prediction. Feature selection is a very important and inevitable part of building a machine learning model. If we select the right features which usually have the most relation with the prediction column, we can achieve high accuracies easily. But if we aren't careful, it may lead to bad consequences.")
    st.write("We will do a chi-squared test on all of the columns to select the 5 most related features.")
    st.info("[Chi-squared tests](https://www.investopedia.com/terms/c/chi-square-statistic.asp#:~:text=Key%20Takeaways%201%20A%20chi-square%20%28%20%CF%872%29%20statistic,of%20freedom%2C%20and%20the%20sample%20size.%20More%20items) are experiments that can show us how strong a relationship a feature has with its prediction column.")
    
    
    col3, col4 = st.columns([3, 5])
    col3.header("Interactive Chi-Squared Test")
    col3.write("Here's an interactive chi-squared test tool that lets you select a column and returns the relationship value between the it and the target column, which is Heart Disease.")
    col3.write("The relationship value is called a 'p-value'. For columns that have a strong relationship, the p-value is usually 0.05 or less.")
    with st.form("2"):
        selected_col = col4.selectbox("Please select a column:", cat_col_dict.values())
        sub1 = st.form_submit_button('Submit To Run Chi-Squared Test')
        if sub1:
            col4.success('Chi-squared Test Performed')
            selected_ = [k for k,v in cat_col_dict.items() if v == selected_col][0]
            res = chi2_contingency(pd.crosstab(df['HeartDiseaseorAttack'], df[selected_]))[0:2]
            st.write("The chi-square stat is:", res[0])
            st.write("The p-value is:", res[1])
            if res[1] <= 0.05:
                st.write("This column is very relevant with the target column.")
		
    st.write('Chi-squared tests are very complicated when it comes to lots of high-quality data like the data we have. As you can see, a chi-squared test returns 2 useful values to us, the p-value and the chi-squared value. Please run the test above and take a look at the outcome. If you ran the test for all of the columns, you should find that some columns have a p-value of 0.0 . This tells us that the columns are associated to a big extent. Which is a good thing.')
    st.write("But, even if we have lots of columns that have strong association with each other, we cannot include all of them. Including lots of columns in a model may cause problems like overfitting and the curse of dimensionality.")
    st.write("[Overfitting](https://www.ibm.com/topics/overfitting) is when your model starts to memorize the training data instead of adapting well to new data. Which causes the model to perform well on the training data but poorly on new, unseen data.")
    st.write("The curse of dimensionality is also a very confusing problem with machine learning models. The term was invented by some computer programmers. The amount of data the model needs for it to reach a certain accuracy gets higher and higher, when you train the model with more columns.")
    st.write("That's why we need to select the most useful columns that have the strongest relation with the target.")
    st.write("So, here is when the chi-square statistic comes into play. When all of the columns have miniscule p-values, and we couldn't judge which column is better, we observe the chi-square value. A bigger chi-squared value means the observed frequencies differ significantly from the expected frequencies, which can happen when there is a strong association between the variables being tested. So, by measuring the chi-squared statistic, we can find the columns we need the most.")

    
    
    st.header("Feature Selection of Numeric Variables")
    st.write("A box plot is effective when it comes to identifying which numeric variables are most strongly related to a category variable, such as Heart Disease or Attack. Numeric variables that have the largest difference in boxes between those with and without Heart Disease or Attack are most likely to be the best predictors for a model.")
    df_long = pd.melt(df.assign(Target = df['HeartDiseaseorAttack'].replace({1:'Yes', 0:'No'})), id_vars='Target', value_vars = num_cols, var_name = 'Feature', value_name = 'Value')
    boxfig = px.box(df_long.assign(Feature = df_long['Feature'].replace(all_col_dict)), x = 'Target', y = 'Value', color = 'Target', title = 'Interactive Box Plot', labels = {'Target':'Heart Disease'}, facet_col_wrap = 2, facet_col = 'Feature', height = 1000)
    boxfig.update_layout(showlegend = False)
    boxfig.update_yaxes(matches=None, title_text = '')
    st.plotly_chart(boxfig)
    st.write("When we are judging according to a box plot whether if a numerical column is suitable for a model, we should consider 4 things.")
    st.write("1. The **Range** of the Box Plot: If the range is too large or too small, it may not be suitable for a model. If the range is too small, the model may not be able to distinguish between different values. If the range is too large, the model may not be able to capture the nuances of the data.")
    st.write("2. The **Outliers**: Look for any outliers in the boxplot. The outliers are dots situated above the upper whisker or below the lower whisker. Outliers can indicate that the data is not normally distributed, which can affect the performance of some models. Additionally, outliers can have a disproportionate effect on the model, skewing the results.")
    st.write("3. The **Skewness**: If the data is highly skewed (meaning that it is not symmetrically distributed), it may not be suitable for a model that assumes a normal distribution, like the naive bayes model.")
    st.write("4. The **Median** and the **Quartiles**: If the median and quartiles are close together, the data may be too homogenous and may not provide enough information for a model.")

    
        
if sideselect == 'Conclusion':
    st.title('Project Conclusion')
    st.write("This project explored Heart Disease and its factors deeply based on a Naive Bayes Classifier. From the chi-squared test we got to know that General Health, Difficulty in Walking, High Blood Pressure and Stroke has the most influence on deciding whether if a person has Heart Disease or not. From the Naive Bayes Classfier, we came to the same conclusion.")
    st.write("According to lots of sources (CDC, HealthLine and Mayo Clinic), the factors I mentioned above are actually very important causes of heart disease. So, this shows that our conclusions fit the advice from prestigious sources.")