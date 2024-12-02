import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import(classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV
import streamlit as st
import io
import seaborn as sns 
import matplotlib.pyplot as plt

uploaded_file = st.sidebar.file_uploader("Please upload your dataset.csv file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, delimiter = ";", quoting=1)
    st.write(" ")
else:
    st.write(" ")


def load_data(file):
    return pd.read_csv(file, delimiter = ";", quoting = 1)

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Model Training", "Chart"])

if "page" not in st.session_state:
    st.session_state["page"] = "Home"


if page == "Home":
    st.title("Machine Learning - White Wine Quality Prediction")
    st.write("Welcome to the White Wine Quality Prediction App!")
    st.write("""
    Upload a dataset and select your machine learning model to predict wine quality.
    """)
elif page == "Data Overview":
    st.title("Data Overview")
    data_info_option = st.sidebar.radio("Data Information", ["Overview", "Dataset Info"])
    if data_info_option == "Overview":
        if uploaded_file is not None:
            st.write("Total rows: {}".format(len(df)))
            st.write("Total columns: {}".format(len(df.columns)))
            st.write(df.head(5))
            st.write(df.describe()) 
        else:
            st.error("Please upload a dataset first.")
    elif data_info_option == "Dataset Info":
        if uploaded_file:
            
            st.write("Dataset Information")
            b = io.StringIO()
            df.info(buf=b)
            info_str = b.getvalue()
            st.write(info_str)
    else:
        st.write("No data uploaded yet.")


elif page == "Model Training":
    st.title("Model Training")
    st.write("Choose a machine learning model and train it on your dataset.")
    
    Variables = st.sidebar.radio("Step 1", ["Select Variables"])
    if Variables == "Select Variables":
        st.write("Select the features (X) and target (y).")
        
        if uploaded_file is None:
            st.error("Please upload a dataset first.")
        else:
            X_col = st.multiselect("Select feature columns (X):", df.columns.tolist())
            y_col = st.selectbox("Select target column (y):", df.columns.tolist())
            if y_col in X_col:
                st.error("Target column (y) cannot be one of the feature columns (X). Please choose distinct columns.")

            elif X_col and y_col:
                
                    
                X = df[X_col].values  # Ensure X is a 2D NumPy array
                y = df[y_col].values  # y can remain 1D
                st.success("Features and target selected successfully!")

                # Create sliders dynamically for each selected feature
                st.write("### Feature Thresholds")
                input_data = {}
                for x_col in X_col:
                    min_value = float(df[x_col].min())
                    max_value = float(df[x_col].max())
                    input_data[x_col] = st.slider(
                        "{}:".format(x_col),
                        min_value=min_value,
                        max_value=max_value,
                        value=min_value,  # Default value
                        step=0.1
                    )
                input_data_df = pd.DataFrame([input_data])
            else:
                st.warning("Please select both features and target columns.")
            
    # Ensure `X` and `y` are defined
    try:
        X, y
    except NameError:
        st.warning("Features (X) and target (y) are not defined. Please select valid columns.")
    else:
        # Proceed to model training only if X and y are defined
        model_choice = st.sidebar.radio("Step 2: Select a Model", ["Random Forest", "SVC", "Logistic Regression"])
        if model_choice:
            st.write(f"### Selected Model: {model_choice}")
            
            # Data split and scaling
            X_train, X_test, y_train, y_test = split_data(X, y)
            
            if model_choice == "Random Forest":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluation metrics
                st.write("Random Forest Results")
                st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred, zero_division=1))
                st.write("Random Forest Cohen's Kappa Score: ", cohen_kappa_score(y_test, y_pred))
                st.write("Random Forest R2 Score: ", r2_score(y_test, y_pred))
                st.write("Random Forest Mean Squared Error: ", mean_squared_error(y_test, y_pred))
                st.subheader("Predicted Wine Quality")
                st.write("The predicted wine quality is:", y_pred)

            elif model_choice == "SVC":
                model = SVC()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluation metrics
                st.write("### SVC Results")
                st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred, zero_division=1))
                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred))
                st.write("SVC Cohen's Kappa Score: ", cohen_kappa_score(y_test, y_pred))
                st.write("SVC R2 Score: ", r2_score(y_test, y_pred))
                st.write("SVC Mean Squared Error: ", mean_squared_error(y_test, y_pred))
                st.subheader("Predicted Wine Quality")
                st.write("The predicted wine quality is:", y_pred)

            elif model_choice == "Logistic Regression":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Evaluation metrics
                st.write("### Logistic Regression Results")
                st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
                st.write("Classification Report:")
                st.write(classification_report(y_test, y_pred, zero_division=1))
                st.write("Logistic Regression Cohen's Kappa Score: ", cohen_kappa_score(y_test, y_pred))
                st.write("Logistic Regression R2 Score: ", r2_score(y_test, y_pred))
                st.write("Logistic Regression Mean Squared Error: ", mean_squared_error(y_test, y_pred))
                st.subheader("Predicted Wine Quality")
                st.write("The predicted wine quality is:", y_pred)

        else:
            st.warning("Please select a model to train.")

elif page == "Chart":
    st.title("Chart")
    st.write("Select a chart type and customize it based on your data.")

    if uploaded_file:
        chart_type = st.sidebar.radio("Chart Type", ["Scatter", "Histogram"])

        if chart_type:
            st.write("### Selected Chart Type: {}".format(chart_type))
            if chart_type == "Scatter":
                st.write("Select the x and y axis variables.")
                x_col = st.selectbox("X Axis:", df.columns.tolist())
                y_col = st.selectbox("Y Axis:", df.columns.tolist())

                # Add threshold slider for x_col
                min_value, max_value = float(df[x_col].min()), float(df[x_col].max())
                selected_threshold = st.slider(
                    "{}:".format(x_col),
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                    step=0.1
                )

                # Check if the same column is selected for both x and y axes
                if x_col == y_col:
                    st.write("Please select different X-axis and Y-axis columns.")
                    st.stop()
                
                elif x_col and y_col:
                    df_filtered = df[(df[x_col] >= selected_threshold[0]) & (df[x_col] <= selected_threshold[1])]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df_filtered[x_col], df_filtered[y_col], color='blue')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title("Scatter Plot of {} vs. {}".format(x_col, y_col))
                    st.pyplot(fig)
                else:
                    st.warning("Please select both X-axis and Y-axis columns.")


            elif chart_type == "Histogram":
                st.write("Select the column to create a histogram.")
                col_to_hist = st.selectbox("Column to create histogram:", df.columns.tolist())

                # Add threshold slider for col_to_hist
                min_value, max_value = float(df[col_to_hist].min()), float(df[col_to_hist].max())
                selected_threshold = st.slider(
                    "{}:".format(col_to_hist),
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                    step=0.1
                )

                if col_to_hist:
                    df_filtered = df[(df[col_to_hist] >= selected_threshold[0]) & (df[col_to_hist] <= selected_threshold[1])]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=df_filtered, x = col_to_hist, ax=ax)
                    ax.set_xlabel(col_to_hist)
                    ax.set_ylabel("Total")
                    st.pyplot(fig)

                else:
                    st.warning("Please select a column to create a histogram.")

    else:
        st.error("Please upload a dataset first.")
