# WEATHER PREDICTION AI
# Overview
The Weather Prediction AI project uses Linear Regression to predict daily average temperatures based on historical weather data. The system is designed for sectors like agriculture, transportation, and disaster management, providing accurate, efficient, and accessible weather forecasting.
# Features
# Accurate Temperature Predictions: Forecasts daily average temperatures for the next 7 days.
# Customizable Input: Users can input a city name and an optional date for tailored temperature predictions.
# Graphical Visualization: Visualizes historical and predicted temperature trends using Matplotlib.
# Error Handling: Provides informative error messages for invalid inputs or unavailable data.
# Linear Regression Explanation: Includes a button for users to learn about how Linear Regression works with a simple example.
# Installation
Clone the repository: git clone https://github.com/Wariha-Asim/Weather-Prediction-Ai.git
Install the required dependencies: pip install -r requirements.txt
# Technologies Used
Python: For implementing Linear Regression, data preprocessing, and UI.
Tkinter: For creating the user interface.
Matplotlib: For generating graphical representations of weather data.
Pandas: For handling and preprocessing the CSV weather data.
Scikit-learn: For implementing the Linear Regression model and evaluation metrics.
# How It Works
Data Preprocessing: Historical weather data (from 2018 to 2022) is processed by converting dates into ordinal values for model training.
Model Training: The model is trained with historical data, where the input (X) is the ordinal date, and the output (y) is the daily temperature.
Prediction: After training, the model predicts the temperature for the next 7 days based on future dates.
Performance Evaluation: The model’s performance is evaluated using Mean Absolute Error (MAE) and R-squared values.
Visualization: The system plots graphs comparing historical data and predicted future temperatures.
# Usage
Input: Enter the city name and an optional specific date to receive predictions.
Output: Get the predicted temperature for the next 7 days along with performance metrics (MAE and R-squared).
Graphical Representation: Visualize both historical and predicted temperatures in separate graphs.
# Contributing
Feel free to fork this repository, create a branch, and submit a pull request. Contributions are welcome to improve the system’s accuracy, add new features, or enhance the user interface.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Conclusion
The Weather Prediction AI system demonstrates the use of machine learning to automate and improve weather forecasting. It provides an accessible and user-friendly interface, offering accurate predictions for daily temperatures and performance evaluation
