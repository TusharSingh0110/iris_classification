import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data function, cached for performance
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

# Load the data
df, target_names = load_data()

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# Sidebar sliders for input features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

# Prepare the input data for prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make the prediction
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# Display the prediction
st.write("Prediction")
st.write(f"The predicted species is: **{predicted_species}**")
