### Titanium Predictor using a Regression Model

## Overview

**Description**

This is the solution given to a Kaggle challenge proposed by a teacher of a master’s degree course.

The objective of this challenge is to predict the concentration of titanium dioxide (titanium TiO2) from various particle and climate measurements taken from different sensors.

The data set contains 9358 instances of average hourly responses from a set of 5 chemical metal oxide sensors integrated into a chemical air quality multi-sensor device.

Attribute Information:
   * 1.	CO_GT: True hourly averaged concentration CO in mg/m^3 (reference analyzer) 
   * 2.	PT08_S1_CO (tin oxide) hourly averaged sensor response (nominally CO targeted) 
   * 3.	NMHC_GT: True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer) 
   * 4.	Nox_GT: True hourly averaged NOx concentration in ppb (reference analyzer) 
   * 5.	PT08_S3_Nox: (tungsten oxide) hourly averaged sensor response (nominally NOx targeted) 
   * 6.	NO2_GT: True hourly averaged NO2 concentration in microg/m^3 (reference analyzer) 
   * 7.	PT08_S4_NO2 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted) 
   * 8.	PT08_S5_O3 (indium oxide) hourly averaged sensor response (nominally O3 targeted) 
   * 9.	T2: Temperature in °C 
   * 10. RH: Relative Humidity (%) 
   * 11. AH: Absolute Humidity
   * 12. PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted) 
 
The missing values are labeled with value -900. 

The goal is to predict the last variable (PT08.S2) from the rest of the measurements.

To achieve this objective, data preprocessing techniques are used and a regression model of Gaussian processes "GPR" is used to estimate since it is the model with which the lowest mean quadratic error "RMSE" was obtained.

The model itself provides an estimate of the uncertainty present in the predictions. During learning the important thing is the choice of a covariance function, known as kernel, since the accuracy of the predictions will depend on the one selected.
It is decided to use the exponential quadratic function as kernel.

To estimate, this model analyzes the training data set and applying Bayesian modeling techniques, generates an estimate using the Maximum a posteriori estimation method "MAP".

## FILES
This project has the following folder distribution: 
  * Readme.md, the file you are currently reading.
  * •	src/ TitaniaPrediction: 
      * code --> In this folder is the file .py with the code.
      * input:
        * data_train.csv - the training set. It contains one sample per row and one variable per column. The last column is the target variable.  
        * data_test.csv - the test set. Contains the input variables. The target variable is lost, you will have to predict it. 





