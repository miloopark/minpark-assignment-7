# Linear Regression Simulation with Hypothesis Testing and Confidence Intervals

This project is a Flask web application that enables users to explore linear regression by generating random data, performing hypothesis testing on the regression slope or intercept, and calculating confidence intervals based on multiple simulations.

## Features

- **Data Generation**: Users can input parameters like intercept, slope, mean, variance, sample size, and number of simulations to generate data points.
- **Scatter Plot and Regression Line**: Visualizes the generated data and fits a linear regression line.
- **Hypothesis Testing**: Allows users to test hypotheses on the slope or intercept with options for greater than, less than, or not equal to tests. Displays the observed statistic, hypothesized value, p-value, and a histogram of simulated statistics.
- **Confidence Intervals**: Calculates confidence intervals for the slope or intercept based on simulated values. Shows a plot of individual simulated estimates, the mean estimate, and the confidence interval range, indicating if the interval includes the true parameter.

## Requirements

- Python 3.7+
- Flask
- numpy
- matplotlib
- scikit-learn
- scipy

## Installation

1. Clone the repository or download the zip file.
2. Navigate to the project directory in your terminal.
3. Install the required packages using pip:

   ```bash
   pip3 install -r requirements.txt
