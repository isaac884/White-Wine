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
import streamlit as st
import io
import seaborn as sns 
import matplotlib.pyplot as plt

##########################################################################################################################################################################################################################################################
# Part 1: Load the dataset
##########################################################################################################################################################################################################################################################

# Function to load the dataset from the uploaded file
def loaddata(file):
    df = pd.read_csv(file, delimiter=";", quoting=1)
    return df


# Use the file uploader to upload the dataset
uploadedfile = st.sidebar.file_uploader("Please upload your dataset.csv file", type="csv")

# Load the dataset in the sidebar if a file is uploaded
if uploadedfile is not None:
    df = loaddata(uploadedfile)
else:
    st.sidebar.warning("Please upload a CSV file to proceed.")

##########################################################################################################################################################################################################################################################
# Part 2: split the data
##########################################################################################################################################################################################################################################################

# Split the dataset into features (X) and target (y)
def splitdata(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


##########################################################################################################################################################################################################################################################
# Part 3: Create the pages for introduction, data overview, model training, and charts
##########################################################################################################################################################################################################################################################

# Add a title for the navigation menu
st.sidebar.title("Navigation")

# Create a menu to switch between pages
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Model Training", "Chart"])

# Initialize session state for page
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Show different pages based on the selected radio button in the sidebar

# Show the Home page
if page == "Home":
    st.title("Machine Learning - White Wine Quality Prediction")
    st.write("Welcome to the White Wine Quality Prediction App!")
    st.write("""
    Upload a dataset and select your machine learning model to predict wine quality.
    """)

# Show the Data Overview page  
elif page == "Data Overview":
    st.title("Data Overview")
    option = st.sidebar.radio("Data Information", ["Overview", "Dataset Info"])
    if option == "Overview":
        if uploadedfile is not None:
            # Show basic information about the dataset
            st.write("Total rows: {}".format(len(df)))
            st.write("Total columns: {}".format(len(df.columns)))
            st.divider()
            st.write("5 Head rows of data")
            st.write(df.head(5)) # Show the first 5 rows
            st.divider()
            st.write("Summary statistics")
            st.write(df.describe()) # Show summary statistics
        else:
            st.error("Please upload a dataset first.")
    elif option == "Dataset Info":
        if uploadedfile:
            # Display dataset information
            st.write("Dataset Information")
            b = io.StringIO() # String b to store info
            df.info(buf=b) # Get info about the dataset and save it in the buffer
            info_str = b.getvalue()# Convert buffer content to string
            st.write(info_str)
    else:
        st.write("No data uploaded yet.")


##########################################################################################################################################################################################################################################################
# Part 4: Model Training Page
##########################################################################################################################################################################################################################################################

elif page == "Model Training":
    
    st.title("Model Training")
    st.write("Choose a machine learning model and train it on your dataset.")
    
    # Step 1: Select the X and Y labels for training first
    Variable = st.sidebar.radio("Step 1", ["Select Variable"])

##########################################################################################################################################################################################################################################################
# let users select variable to X and y, they can select any variable X depends on them 

    if Variable == "Select Variable":
        st.write("Select the features (X) and target (y).")
        
        if uploadedfile is None:
            st.error("Please upload a dataset first.")
        else:
            # show and select from the variable list
            X_c = st.multiselect("Select feature columns (X):", df.columns.tolist())
            y_c = st.selectbox("Select target column (y):", df.columns.tolist())
            if y_c in X_c:
                st.error("Target column (y) cannot be one of the feature columns (X). Please choose distinct columns.")
            elif X_c and y_c:
                
                    
                X = df[X_c].values  # Ensure X is a 2D NumPy array
                y = df[y_c].values  # y can remain 1D
                st.success("Features and target selected successfully!")

                # Create sliders, they and select any input values
                st.write("### Feature Input")
                st.write("You can though this slider to assume any input values")
                input_data = {}
                for x_c in X_c:
                    min_value = float(df[x_c].min())
                    max_value = float(df[x_c].max())
                    input_data[x_c] = st.slider(
                        "{}:".format(x_c),
                        min_value=min_value,
                        max_value=max_value,
                        value=min_value, 
                        step=0.1
                    )

                # Convert the input_data dictionary to a DataFrame
                input_df = pd.DataFrame([input_data])
            else:
                # if haven't selected X and Y, then show the warning message
                st.warning("Please select both features and target columns.")
   
    #  defined `X` and `y`
    try:
        X, y # Check if X and y exist
    except NameError:
        st.warning("Features (X) and target (y) are not defined. Please select valid columns.")
     
    else:
        # Let the user choose a machine learning model
        choice = st.sidebar.radio("Step 2: Select a Model", ["Random Forest", "SVC", "Logistic Regression"])
        if choice:
            st.write("### Selected Model: {}".format(choice))
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = splitdata(X, y)
            
##########################################################################################################################################################################################################################################################
# Random Forest page
            if choice == "Random Forest":
                RCm = RandomForestClassifier()
                # Train the Random Forest model using training data
                RCm.fit(X_train, y_train)
                # Make predictions on the test data
                y_pred = RCm.predict(X_test)

                # Display the accuracy score and Cohen's Kappa Score use by metric
                st.metric(label="Accuracy Score", value=accuracy_score(y_test, y_pred))
                st.write("")
                st.metric(label="Logistic Regression Cohen's Kappa Score", value= cohen_kappa_score(y_test, y_pred))
                st.write("")

                # Add a horizontal divider for better visual separation
                st.divider()

                # Use the expandable section to show the classification report
                st.subheader("Classification Report")
                with st.expander("Classification Report"):
                    st.write(classification_report(y_test, y_pred, zero_division=1))
                st.write("")
                st.divider()

                # Use the expandable section to show the confusion matrix
                # Create a heatmap to visualize the confusion matrix
                st.subheader("Confusion Matrix")
                with st.expander("Confusion Matrix"):
                    RCmatrix = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(RCmatrix, annot=True, fmt="d", cmap="Blues", ax=ax) 
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                st.write("")
                st.divider()
                # Predict the wine quality based on the input data
                st.subheader("Predicted Wine Quality")
                pre = RCm.predict(input_df.values)
                st.write("The predicted wine quality is:", pre)
                
##########################################################################################################################################################################################################################################################
# SVC page
            elif choice == "SVC":
                SVCm = SVC()
                # Train the Random Forest model using training data
                SVCm.fit(X_train, y_train)
                # Make predictions on the test data
                y_pred = SVCm.predict(X_test)

                
                st.metric(label="Accuracy Score", value=accuracy_score(y_test, y_pred))
                st.metric(label="Logistic Regression Cohen's Kappa Score", value= cohen_kappa_score(y_test, y_pred))
                st.write("")
                st.divider()

                st.subheader("Classification Report: ")
                with st.expander("Classification Report:"):
                    st.write(classification_report(y_test, y_pred, zero_division=1))
                st.write("")
                st.divider()
                
                st.subheader("Confusion Matrix")
                with st.expander("### Confusion Matrix"):
                    SVCmatrix = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(SVCmatrix, annot=True, fmt="d", cmap="red", ax=ax) 
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                st.write("")
                st.divider()

                st.subheader("Predicted Wine Quality")
                p = SVCm.predict(input_df.values)
                st.write("The predicted wine quality is:", p)

##########################################################################################################################################################################################################################################################
# Logistic Regression page 
            elif choice == "Logistic Regression":
                LRm = LogisticRegression(max_iter=500)
                # Train the Random Forest model using training data
                LRm.fit(X_train, y_train)
                # Make predictions on the test data
                y_pred = LRm.predict(X_test)

                st.metric(label="Accuracy Score", value=accuracy_score(y_test, y_pred))
                st.metric(label="Logistic Regression Cohen's Kappa Score", value= cohen_kappa_score(y_test, y_pred))
                st.write("")
                st.divider()

                st.subheader("Classification Report: ")
                with st.expander("Classification Report:"):
                    st.write(classification_report(y_test, y_pred, zero_division=1))
                st.write("")
                st.divider()
                
                st.subheader("Confusion Matrix")
                with st.expander("### Confusion Matrix"):
                    LRmatrix = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(LRmatrix, annot=True, fmt="d", cmap="red", ax=ax) 
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                st.write("")
                st.divider()

                st.subheader("Predicted Wine Quality")
                pr = LRm.predict(input_df.values)
                st.write("The predicted wine quality is:", pr)
                
        else:
            st.warning("Please select a model to train.")

##########################################################################################################################################################################################################################################################
# Part 5: Chart page
##########################################################################################################################################################################################################################################################

elif page == "Chart":
    st.title("Chart")
    st.write("Select a chart type and customize it based on your data.")

    if uploadedfile:
        chart = st.sidebar.radio("Chart Type", ["Scatter", "Histogram"])

        if chart:
            st.write("### Selected Chart Type: {}".format(chart))

##########################################################################################################################################################################################################################################################
# Scatter page
            if chart == "Scatter":
                
                st.write("Select the x and y axis variable.")
                # Let the user to select the x and y axes for the scatter plot
                # show and select from the variable list
                x_c = st.selectbox("X Axis:", df.columns.tolist())
                y_c = st.selectbox("Y Axis:", df.columns.tolist())

                # Add threshold slider for x_c
                min_value, max_value = float(df[x_c].min()), float(df[x_c].max())
                select = st.slider(
                    "{}:".format(x_c),
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                    step=0.1
                )

                # Check if the same column is selected for both x and y axes
                if x_c == y_c:
                    st.write("Please select different X-axis and Y-axis columns.")
                    st.stop()
                # Check if valid X and Y  are selected
                elif x_c and y_c:
                    # Filter the DataFrame based on the selected threshold for the X-axis
                    df_f = df[(df[x_c] >= select[0]) & (df[x_c] <= select[1])]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df_f[x_c], df_f[y_c], color='blue')
                    ax.set_xlabel(x_c)
                    ax.set_ylabel(y_c)
                    ax.set_title("Scatter Plot of {} vs. {}".format(x_c, y_c))
                    st.pyplot(fig)
                else:
                    st.warning("Please select both X-axis and Y-axis columns.")

##########################################################################################################################################################################################################################################################
# Histogram page 
            elif chart == "Histogram":
                st.write("Select the column to create a histogram.")
                col = st.selectbox("Column to create histogram:", df.columns.tolist())

                # Add threshold slider for col
                min_value, max_value = float(df[col].min()), float(df[col].max())
                select = st.slider(
                    "{}:".format(col),
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                    step=0.1
                )

                if col:
                    df_f = df[(df[col] >= select[0]) & (df[col] <= select[1])]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data=df_f, x = col, ax=ax)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Total")
                    st.pyplot(fig)

                else:
                    st.warning("Please select a column to create a histogram.")

    else:
        st.error("Please upload a dataset first.")
###########################################################################################################################################################################################################################################################       
