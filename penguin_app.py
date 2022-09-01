import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)
df.head()
df = df.dropna()
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})
df['sex'] = df['sex'].map({'Male':0,'Female':1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})

X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


#SVC model
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

#LogisticRegression model.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

#RandomForestClassifier model.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache()
def prediction(model , island , bill_length_mm , bill_depth_mm , flipper_length_mm , body_mass_g , sex):
    species = model.predict([[island , bill_length_mm , bill_depth_mm , flipper_length_mm , body_mass_g , sex]])
    species = species[0]
    if species == 0:
        return 'Adelie'
    elif species == 1:
        return 'Chinstrap'
    else:
        return 'Gentoo'
    
    
st.title("Penguin Species Prediction App")  
bill_length = st.sidebar.slider("Bill Length", float(df['bill_length_mm'].min()) , float(df['bill_length_mm'].max()))
bill_depth = st.sidebar.slider("Bill Depth", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
flipper_length = st.sidebar.slider("Flipper Length", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
body_mass = st.sidebar.slider("Body Mass(grams)", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))

sex = st.selectbox("Sex", ("Male", "Female")) 
if sex == "Male":
	sex = 0
else:
	sex = 1
    
island = st.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
if island == "Biscoe":
	island = 0
elif island == "Dream":
	island = 1
else:
	island = 2
    
    
classifier = st.sidebar.selectbox('Classifier' , ('Support Vector Machine' , 'RandomForestClassifier' , 'Logistic Regression'))

if st.sidebar.button("Predict"):
    if classifier == 'Support Vector Machine':
        species_type = prediction(svc_model , bill_length , bill_depth , flipper_length , body_mass , sex , island) 
        score = svc_model.score(X_train , y_train)
    elif classifier == 'RandomForestClassifier':
        species_type = prediction(rf_clf , bill_length , bill_depth , flipper_length , body_mass , sex , island)
        score = rf_clf.score(X_train , y_train)
    else:
        species_type = prediction(log_reg , bill_length , bill_depth , flipper_length , body_mass , sex , island)
        score = log_reg.score(X_train , y_train)
    
    st.write("Species predicted:", species_type)
    st.write("Accuracy score of this model is:", score)