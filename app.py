import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset and pre-process
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/LENOVO/Desktop/ML/KNN/Social_Network_Ads.csv")
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    sc = StandardScaler()
    df.iloc[:, 1:-1] = sc.fit_transform(df.iloc[:, 1:-1])
    return df, le, sc

df, le, sc = load_data()
X = df.iloc[:, [1,2,3]].values
y = df.iloc[:, -1].values

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
classifier = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
classifier.fit(X_train, y_train)

# Streamlit app
st.title("Social Network Ads Predictor")

st.write("""
### Input the details for prediction
""")

# User input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=10, max_value=100)
estimated_salary = st.slider("Estimated Salary", min_value=15000, max_value=150000)

# Making predictions
if st.button("Predict"):
    input_data = pd.DataFrame([[gender, age, estimated_salary]], columns=["Gender", "Age", "EstimatedSalary"])
    input_data['Gender'] = le.transform(input_data['Gender'])
    input_data = sc.transform(input_data)
    prediction = classifier.predict(input_data)
    
    st.write(f"Prediction: Purchased: {prediction[0]}")

# Model accuracy
y_pred = classifier.predict(X_test)
ac = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {ac}")

# Confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,7))  # Create a figure and a set of subplots
sns.heatmap(cm, annot=True, ax=ax)  # Use the 'ax' argument to specify the subplot
ax.set_xlabel('Predicted')
ax.set_ylabel('Truth')
st.pyplot(fig)  # Pass the figure object to st.pyplot
