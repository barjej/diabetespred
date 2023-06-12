import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout='wide',page_title='Diabetes Diagnostic Predict')

df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRW-2rCgMoo9glcRfoAAZHUIgQpyb4sXv9DATBsQoihLtBbCo-p_BUwfrNcaq8OCCOe2ZhS0mVOvkR-/pub?gid=1879661496&single=true&output=csv')

#########################################################################
#DATA PREPROCESSING

#Handling Outlier
df['Pregnancies']=df.Pregnancies.mask(df.Pregnancies == 0, df['Pregnancies'].mean())
df['Pregnancies']=df.Pregnancies.mask(df.Pregnancies > 12, df['Pregnancies'].mean())
df['Glucose']=df.Glucose.mask(df.Glucose == 0, df['Glucose'].mean())
df['BloodPressure']=df.BloodPressure.mask(df.BloodPressure == 0, df['BloodPressure'].mean())
df['BloodPressure']=df.BloodPressure.mask(df.BloodPressure < 40, df['BloodPressure'].mean())
df['BloodPressure']=df.BloodPressure.mask(df.BloodPressure > 105, df['BloodPressure'].mean())
df['BMI']=df.BMI.mask(df.BMI == 0, df['BMI'].mean())
df['BMI']=df.BMI.mask(df.BMI > 48, df['BMI'].mean())

#Select column based on correlation
df = df.drop(['SkinThickness','Insulin','BloodPressure','DiabetesPedigreeFunction'],axis=1)

#machinelearning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
y = df.iloc[:,-1]
x = df.iloc[:,0:-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=0)
#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
#########################################################################
#FRONTEND

st.title('Diabetes Diagnostic Predict')

st.dataframe(df, use_container_width=True)

diabetespositive = df['Outcome'].loc[df['Outcome'] == 1].count()
percpositive = df['Outcome'].loc[df['Outcome'] == 1].count()/df['Outcome'].count()
diabetesnegative = df['Outcome'].loc[df['Outcome'] == 0].count()
percnegative = df['Outcome'].loc[df['Outcome'] == 0].count()/df['Outcome'].count()

modemode = df.loc[df['Outcome'] == 1]
modeage = modemode['Age'].mode().values[0]
avgglu = df['Glucose'].loc[df['Outcome'] == 0].mean()

mx_positive, mx_negative, mx_age, mx_glucose = st.columns(4)

with mx_positive:
    st.metric(
        "Positive Diabetes (Amount)",
        value= diabetespositive
    )
with mx_negative:
    st.metric(
        "Negative Diabetes (Amount)",
        value= diabetesnegative
    )
with mx_age:
    st.metric(
        "Most Positive Diabetes (Age)",
        value= f'{modeage} years old'
    )
with mx_glucose:
    st.metric(
        "Average Good Glucose Rate",
        value= round(avgglu,2),
    )

st.caption('Facts : The most influential variable in the process of making this project is the Glucose variable, which means that Glucose is the main factor causing Diabetes itself.')

st.header('Diabetes Graph')

agecount = df.groupby(['Age','Outcome']).size().reset_index(name='count')
agecount['Outcome'] = agecount['Outcome'].replace(0,'Negative').replace(1,'Positive')
custom_colors = alt.Scale(domain=['Negative', 'Positive'], range=['#38B000', '#E63946'])

diabetesbar = alt.Chart(agecount).mark_bar().encode(
        alt.X('Age',title="Age"),
        alt.Y('count',title='Amount'),
        alt.Color('Outcome',scale=custom_colors),
    )

st.altair_chart(diabetesbar, use_container_width=True)

st.header('Diabetes Graph per Age')

ages = df['Age'].unique()

ageselect = st.selectbox("Age (based on data)",sorted(ages))

basedperage = agecount.loc[agecount['Age'] == int(ageselect)]

diabetesbarperage = alt.Chart(basedperage).mark_bar().encode(
        alt.X('Age',title="Age"),
        alt.Y('count',title='Amount'),
        alt.Color('Outcome',scale=custom_colors),
        alt.Column('Outcome', header=alt.Header(title='Outcome'))
    ).properties(
        width=100,
        height=200
    )

st.altair_chart(diabetesbarperage)

st.header('Diabetes Detection Tool (Diagnose)')

import numpy as np
preg = st.number_input('How many times pregnant : ', 0,20)
glu = st.number_input('Glucose level in blood : ',0,300)
bmi = st.number_input('Body Mass Index (BMI) : ',0,50)
age = st.number_input('Age : ',0,120)

try:
    pregt = float(preg)
    glut = float(glu)
    bmit = float(bmi)
    aget = int(age)
    demo = np.array([pregt,glut,bmit,aget])
    demo = demo.reshape(1,-1)
    pred = lr.predict(demo)
    pred = pred.reshape(-1,1)
    if st.button('Enter') : 
        if pred[0] == 0:
            st.write('You are NEGATIVE diabetes')
        elif pred[0] == 1:
            st.write('You are POSITIVE diabetes')
    else:
        st.write('')
except ValueError:
    st.write("Please input your data correctly!")

from sklearn.metrics import accuracy_score

st.write("Model Accuracy (Logistic Regression) :\n",round(accuracy_score(y_test,y_pred_lr),2))

st.caption('Important : This model is a DIAGNOSTIC tool for detecting diabetes only. It is recommended to consult to a doctor to get more accurate results')
