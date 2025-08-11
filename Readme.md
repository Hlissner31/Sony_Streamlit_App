
# Streamlit App README

## Overview

This repository contains several interactive Streamlit applications designed for different types of analysis and forecasting, including predicting high-value content, survival analysis, ARIMA time-series forecasting, and sentiment prediction.

### Apps in this Repository

1. **High Value Content Classifier**  
   - A content ranking and group classification app designed to predict the likelihood that a piece of content belongs to the "high-value" bin, which represents content with the highest engagement and revenue potential. The app uses **Random Forest** and **XGBoost** models, trained on a dataset from the Sony sales team, to predict content value based on features such as video duration, user rating, distributor, IP type, title type, and genre.  
   - [app link](#)

2. **ARIMA Time Series Forecasting Dashboard**  
   - A dashboard for time-series forecasting using ARIMA models. Users can upload a dataset, select a channel, and generate ARIMA forecasts with validation metrics.  
   - [app link](#)

3. **YouTube Title Impact Predictor (Sentiment App)**  
   - This app predicts the potential views and revenue for a given YouTube title based on various factors such as sentiment, length, and genre.  
   - [app link](#)

4. **Sony YouTube Survival Analysis**  
   - This app uses survival analysis techniques to explore how long videos stay relevant based on when daily views fall below 1% of total views.  
   - [app link](#)

---

## High Value Content Classifier (`Predapp.py`)

### Overview

The **High Value Content Classifier** app was designed to offer an interactive, user-friendly interface that predicts the likelihood of a piece of content belonging to the "high-value" bin, which represents content with the highest engagement and revenue potential. The app uses two machine learning models—**Random Forest** and **XGBoost**—trained on a dataset provided by the Sony sales team. The primary goal is to rank content by its predicted viewership, focusing on identifying the top-performing content in the **4th bin**, which is vital for content selection.

### How the Prediction of the 4th "High-Value" Bin is Made

The app predicts the likelihood that a piece of content will fall into the **4th bin**, which represents the highest-value content based on engagement and viewership. 

- **Random Forest**: Uses an ensemble of decision trees, with each tree trained independently on different subsets of data. The final prediction is made by averaging the percentage of trees that classify the content as belonging to the 4th bin.
- **XGBoost**: A boosting model that improves upon previous trees by correcting errors. XGBoost calculates the probability of content being classified into each bin, with the 4th bin probability as the final output.

Once predictions are obtained from both models, they are averaged to generate a final prediction for the likelihood of the content being high-value. For example, if Random Forest predicts a 70% likelihood and XGBoost predicts 80%, the final prediction is 75%.

### App Workflow

- **Input**: Users input key details like video duration, user rating, distributor, IP type, title type, and genre.
- **Prediction**: The inputs are transformed into a feature vector and passed through the models. The likelihood of the content belonging to the high-value 4th bin is displayed as a percentage.
- **Bulk Predictions**: Users can upload a CSV or Excel file, and the app preprocesses the data, generates predictions for multiple entries, and allows users to download the results in an Excel file.

### Manual Cluster Assignment for New Content

The app also includes a **manual cluster assignment** feature, which assigns new content to clusters based on features like video duration, user rating, and genre. This helps categorize content based on its similarity to clusters derived from historical performance data. Although the current cluster assignment is relatively accurate, it can be further refined as more historical performance data for new content becomes available.

### Benefits

- **Real-Time Predictions**: Predict the likelihood of a piece of content being high-value.
- **Batch Predictions**: Upload multiple content entries for predictions at once.
- **Cluster Assignment**: Categorize new content based on available features, with room for future refinement as more data is gathered.

---

## ARIMA Time Series Forecasting Dashboard (`arima_dashboard.py`)

### Overview

- **Upload Dataset**: Users can upload a CSV file containing date and viewership data.
- **Channel Selection**: Choose a specific channel to analyze.
- **ARIMA Forecasting**: Automatically trains ARIMA models and provides forecasts for future views.
- **Validation Metrics**: RMSE, MAE, MAPE to evaluate forecast performance.
- **Stationarity Test**: Augmented Dickey-Fuller test to check for stationarity.
- **Kaplan-Meier Plot**: Displays survival analysis for video relevance.

---

## YouTube Title Impact Predictor (`Sentiment_app.py`)

### Overview

- **Input**: Users input a YouTube title.
- **Sentiment & Title Features**: Sentiment score, title length, and clickbait detection.
- **Prediction**: Predict estimated views and revenue based on a trained model.
- **Feedback**: Interpretive feedback based on title features.
- **Feature Importance**: Visualize the importance of different features for revenue prediction.

---

## Sony YouTube Survival Analysis (`Surival_Analysis_Sony.py`)

### Overview

- **Data Upload**: Upload daily viewership data.
- **Survival Analysis**: Kaplan-Meier curve to estimate the survival function of content (how long it remains relevant).
- **Median Survival**: Displays median survival times by channel and genre.
- **Download Results**: Allows users to download retention group data as Excel files.

---

## Installation

### Prerequisites

- Python 3.x
- Streamlit
- Required libraries: pandas, numpy, matplotlib, seaborn, lifelines, scikit-learn, joblib, pickle, statsmodels

### Setup Instructions

1. Clone this repository or download the code files.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. To run any of the apps, navigate to the respective Python file and run it via Streamlit:

   ```bash
   streamlit run <app_filename>.py
   ```

---

## File Uploads

- **Input File Formats**: 
  - For the **ARIMA dashboard** and **YouTube Title Impact Predictor**, users should upload a CSV file.
  - For the **High Value Content Classifier**, both CSV and Excel files are accepted.
  
- **Sample Data**: Sample data is available for testing purposes. Ensure the data format matches the expected structure in the respective apps.

---

## Usage Instructions

1. **High Value Content Classifier**
   - Enter video features in the sidebar or upload a CSV file.
   - Click "Predict High Value Content Likelihood" to get the content's classification probability.
   - Download batch predictions in Excel format.

2. **ARIMA Time Series Forecasting Dashboard**
   - Upload your CSV with the necessary columns: `date`, `channel_display_name`, and `views`.
   - Select a channel and configure model parameters like AR terms, differencing, and MA terms.
   - Click "Run ARIMA Analysis" to generate forecasts.

3. **YouTube Title Impact Predictor**
   - Enter a YouTube title and click "Predict" to estimate views and revenue.
   - View interpretive feedback and suggested improvements for the title.

4. **Sony YouTube Survival Analysis**
   - Upload a CSV with daily viewership data.
   - Apply filters for channel and genre, and view the Kaplan-Meier survival curve.
   - Download retention group data as Excel.

---

## Known Issues and Limitations

- Ensure the input file is correctly formatted to avoid parsing issues.
- The **High Value Content Classifier** may require adjusting feature names to match the training model.
- The **Sony YouTube Survival Analysis** may display incomplete data if the input file has missing or incorrect columns.

---

## Conclusion

This repository provides a range of interactive tools for analyzing YouTube data, predicting content performance, and conducting survival analysis. These tools leverage machine learning models and statistical methods, offering valuable insights for content creators, analysts, and marketers.
