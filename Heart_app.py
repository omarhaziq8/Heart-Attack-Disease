# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:40:07 2022

@author: pc
"""

import pickle 
import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.getcwd(),'Best_Model.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)


#%% Data Loading

CSV_PATH = os.path.join(os.getcwd(),'heart.csv')
df = pd.read_csv(CSV_PATH)

#%% Header
st.title('Heart Attack Predictor :wave:')

#%% Training Data
st.subheader('KL Heart Data :scream:')
st.write(df.describe())

#%% Image
st.title('Do you feel pain in the chest? this is the true broken heart means!! Start to check by key in your details! :broken_heart:')
image = Image.open('CVD.jpg')
st.image(image, caption='CardioVascular Disease')

#%% Visualisation

st.subheader('To show that these features have high correlation to the HeartAttack Disease from dataset :computer:')
fig,axes = plt.subplots(2,2,sharex=False,figsize=(10,10))
fig.suptitle('Distribution of data')
# share the same x axis equal to true: it will become no distribution/small range
# in weight and height, while if it is false they produce largest range~!

axes[0,0].set_title('Distribution of age')
sns.distplot(df['age'],ax=axes[0,0])

axes[1,0].set_title('Distribution of thalachh')
sns.distplot(df['thalachh'],ax=axes[1,0])

axes[0,1].set_title('Distribution of oldpeak')
sns.distplot(df['oldpeak'],ax=axes[0,1])

axes[1,1].set_title('Distribution of thall')
sns.countplot(df['thall'],ax=axes[1,1])

st.pyplot(fig)

#%% Form predictor 

# x = df.loc[:,['age','thalachh','oldpeak','thall']] # Features
# x_new = [37,187,3.5,2]

with st.form("Patient's info"):
    st.write("Please enter your details and get to know the results :grin:")
    
    age = int(st.number_input('Key in your age'))
    thalachh = int(st.number_input('Key in your thalachh'))
    oldpeak = int(st.number_input('Key in your oldpeak'))
    thall = int(st.number_input('Key in your thall'))
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("age", age, "thalachh", thalachh, "oldpeak", oldpeak,
                 "thall", thall)
        temp = np.expand_dims([age,thalachh,oldpeak,thall],axis=0)
        outcome = model.predict(temp)
        
        outcome_dict = {0: 'Less risk of getting Heart Attack',
                        1: 'Higher chance of getting Heart Attack'}
        
        st.write(outcome_dict[outcome[0]]) # outcome is an array, 0 is slicing
        
        if outcome == 1:
            st.write('Please take care of your health bro!')
        else:
            st.balloons()
            st.write('Good :thumbsup:')











