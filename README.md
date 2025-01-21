# Salary_LinearRegression
This project implements a simple linear regression model to predict employee salaries based on years of experience. 
The model is developed from scratch using Python and foundational mathematical concepts without relying on machine learning libraries like Scikit-learn or TensorFlow. The focus is on understanding the underlying mechanics of linear regression, including gradient descent optimization, cost function computation, and parameter tuning.

The project includes:

1) Data visualization to explore the relationship between years of experience and salary.
2) A cost function implementation to measure model accuracy.
3) Gradient descent optimization to minimize the cost and improve predictions.
4) Visualization of the optimization process and the final regression line.

**Cost**:
The cost function measures how well the linear regression model fits the data by quantifying the difference between predicted and actual values. Minimizing the cost ensures better predictions.

**Gradient**:
The gradient indicates the direction and rate of change of the cost with respect to the model parameters (w and b). It helps determine how the parameters should be adjusted to reduce the cost.

**Iterative Gradient (Gradient Descent)**:
Gradient descent is an iterative optimization algorithm that uses the gradient to update the model parameters step by step. This process gradually minimizes the cost function, leading to optimal parameters for the linear regression model.

**Features**

-Data Loading: Reads salary data from a CSV file.

-Data Visualization: Displays scatter plots to illustrate the relationship between the independent and dependent variables.

-Cost Function Implementation: Calculates the mean squared error to assess model performance.

-Gradient Descent: Optimizes model parameters (w and b) through iterative updates.

-Prediction and Visualization: Plots the final regression line alongside the training data.

**Dataset**

The dataset used for this project is sourced from Kaggle: [Salary Dataset for Simple Linear Regression](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression/data).
It consists of two columns:

-Years of Experience: Independent variable (input feature).

-Salary: Dependent variable (target).

The dataset is loaded from a CSV file (salary_dataset.csv) and preprocessed for training.

**How to Use**
Just clone repository (https://github.com/rubino1996/Salary_LinearRegression.git) and run the main.py function 
