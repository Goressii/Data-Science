import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu
from sklearn import preprocessing

def load_dataset(dataset_file):
    data = pd.read_csv(dataset_file)
    return data

def main():
    st.title("Second homework")
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = load_dataset(uploaded_file)
        st.dataframe(data)
    
        columns_list = data.columns
        first_option = st.selectbox("Select the first column", columns_list)
        second_option = st.selectbox("Select the second column", columns_list)
        selected_data = data[[first_option, second_option]]
        
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(
            go.Histogram(x=data[first_option]),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=data[second_option]),
            row=1, col=2
        )
        
        fig.update_layout(height=600, width=800, title_text="Column distribution")
        st.plotly_chart(fig)
        algorithm_list = ["ANOVA", "Mann-Whitney U-test (numeric only)"]
        algorithm = st.selectbox("Select a hypothesis test algorithm", algorithm_list)
        
        if algorithm == "ANOVA":
            if data[first_option].dtype == "float64" or data[first_option].dtype == "int64":
                statistics, pvalue = f_oneway(data[first_option], data[second_option])
            else:
                label_encoder = preprocessing.LabelEncoder()
                selected_data[first_option] = label_encoder.fit_transform(selected_data[first_option])
                selected_data[second_option] = label_encoder.fit_transform(selected_data[second_option])
                statistics, pvalue = f_oneway(selected_data[first_option], selected_data[second_option])
            
        elif algorithm == "Mann-Whitney U-test (numeric only)":
            statistics, pvalue = mannwhitneyu(selected_data[first_option], selected_data[second_option])
            
        col1, col2 = st.columns(2)
        col1.metric("Statistics", f"{statistics:.{5}f}")
        col2.metric("p-value", f"{pvalue:.{5}f}")


        
    
main()