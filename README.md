# Rinex-Project-1-LinearRegression-Advertising

# Linear Regression Model on Advertising Data

This Python script demonstrates how to perform a linear regression analysis on a dataset containing advertising spend data. The goal is to predict sales (Sales) based on three different features: TV, Radio, and Newspaper advertising budgets.

The script proceeds through the following steps:

✔ Load and examine the data.

✔ Visualize relationships between the features and target variable (Sales).

✔ Prepare the data for training a linear regression model.

✔ Train the linear regression model and evaluate its performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

✔ Refit the model using a subset of features (TV and Radio) and evaluate its performance.

# Libraries Used

Pandas: Data manipulation and analysis.

Seaborn: Visualization for plotting pairplots.

Matplotlib: For inline plotting.

Scikit-Learn: For machine learning tools, including linear regression and model evaluation.

NumPy: For numerical operations like calculating RMSE.

# Steps in the Script

# 1.Loading Data:

python

Copy

data = pd.read_csv("Advertising.csv", index_col=0)

The dataset Advertising.csv is loaded into a pandas DataFrame. The first column (index) is set as the index column.

# 2.Data Exploration:

python

Copy

data.head()

data.tail()

data.shape

head() and tail() display the first and last five rows of the dataset, respectively, to give an overview of the data.

shape outputs the number of rows and columns in the dataset.

# 3.Visualization:

python

Copy

sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, aspect=0.7, kind='reg')

A pairplot is created to visualize the relationship between the independent variables (TV, Radio, Newspaper) and the dependent variable (Sales). The kind='reg' argument adds regression lines to each scatter plot.

# 4.Preparing Features and Target Variables:

python

Copy

feature_cols = ['TV', 'Radio', 'Newspaper']

X = data[feature_cols]

y = data['Sales']

The independent variables (features) are selected as TV, Radio, and Newspaper.

The dependent variable (Sales) is set as y.

# 5.Train-Test Split:

python

Copy

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

The data is split into training and testing sets (75% training, 25% testing) using train_test_split from sklearn.model_selection.

# 6.Training the Linear Regression Model:

python

Copy

linreg = LinearRegression()

linreg.fit(X_train, y_train)

A linear regression model is created and fitted to the training data.

# 7.Model Coefficients:

python

Copy

print(linreg.intercept_)

print(linreg.coef_)

The intercept and coefficients (weights for each feature) of the trained linear regression model are printed.

# 8.Model Prediction:

python

Copy

y_pred = linreg.predict(X_test)

The trained model is used to make predictions on the test data.

Evaluation: The script computes different error metrics:

# 9.Mean Absolute Error (MAE):

python

Copy

print(metrics.mean_absolute_error(true, pred))

MAE represents the average of absolute differences between predicted and actual values.

# 10.Mean Squared Error (MSE):

python

Copy

print(metrics.mean_squared_error(true, pred))

MSE measures the average squared difference between predicted and actual values.

# 11.Root Mean Squared Error (RMSE):

python

Copy

print(np.sqrt(metrics.mean_squared_error(true, pred)))

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

RMSE gives the square root of MSE and is a commonly used evaluation metric to measure the magnitude of error.

# 12.Model Refit with Subset of Features:

python

Copy

feature_cols = ['TV', 'Radio']

X = data[feature_cols]

The model is refitted using only TV and Radio as features to observe the impact of excluding Newspaper.

The training and testing process is repeated with the new feature set, and evaluation is done using RMSE.

# Key Outputs

Linear Regression Coefficients:
The model outputs the intercept and coefficients for the features used in the linear regression model.

Evaluation Metrics:
The Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are calculated for the model predictions.

Model Performance:
The error metrics help evaluate the performance of the regression model in predicting sales.

# Conclusion

This script demonstrates the application of linear regression to predict sales based on advertising spend in various media channels. The evaluation metrics show how well the model fits the data, and the refitting with fewer features highlights the impact of feature selection.

# Requirements

Python 3.x

Libraries: pandas, seaborn, matplotlib, scikit-learn, numpy

You can install the necessary libraries using the following command:

bash

Copy

pip install pandas seaborn matplotlib scikit-learn numpy

Notes

✔ The dataset file Advertising.csv must be in the same directory as the script or provide the full path to the file.


