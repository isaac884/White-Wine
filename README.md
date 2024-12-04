# White Wine Quality Prediction 

## About
-This app helps users predict the quality of white wine using machine learning models.  
-Upload the `winequality-white.csv` file to train different models and make predictions.

## Illstruate
- Upload the `winequality-white.csv` dataset and use the "Overview" page to explore the data.
- Train machine learning models, including:
  - Random Forest
  - Support Vector Machine (SVC)
  - Logistic Regression
- View performance metrics such as accuracy, classification report, Cohen's Kappa score, R2 score, mean squared error, and confusion matrix.
- Visualize the data with scatter plots and histograms.

## How to Use
1. **Install Requirements**  
   Ensure Python is installed on your system. Then, install the required libraries:
   ```bash:
   pip install streamlit
   pip install scikit-learn
   pip install pandas
   pip install matplotlib
   pip install numpy
   pip install seaborn
   ```

2. **Install Dataset**  
   Download the `winequality-white.csv` file.

3. **Run the App**  
   Download the Python script and run the app using Streamlit:
   ```bash:
   streamlit run White_Wine.py
   ```

4. **Upload Dataset**  
   Upload the `winequality-white.csv` file in the web app. 

5. **Overview Data**  
   Navigate to the "Data Overview" page to check the dataset’s summary and information.

6. **Training Models**  
   - Select the feature columns (`X`) and the target column (`y`).
   - Choose a machine learning model and view its results.

7. **Visualize Data**  
   Use the "Chart" page to create scatter plots or histograms of the data.

## Dataset
This app is designed to work with datasets like `winequality-white.csv`, which contains columns like acidity, alcohol, and the target column for wine quality.

## Libraries
This project uses the following Python libraries:
- `pandas` and `numpy` for data manipulation.
- `sklearn` for machine learning models and metrics.
- `streamlit` for building the web interface.
- `seaborn` and `matplotlib` for data visualization.

## Files
- `White_Wine.py`: Main script for the app.
- `requirements.txt`: A list of required Python libraries.
- `winequality-white.csv`: The dataset used for prediction.
- `Procfile`: For deployment (e.g., on Heroku).
