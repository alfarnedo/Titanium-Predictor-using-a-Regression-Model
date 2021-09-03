# -*- coding: utf-8 -*-
#******************************************* LIBRARIES TO USE *************************************

import pandas as pd      # The necessary functions for data analysis are imported.
import numpy as np       # The necessary functions for mathematical computing are imported.
from scipy import spatial                       # The necessary functions for space computing are imported.
import csv                                      # The functions are imported for the processing of files .csv.
from sklearn.metrics import mean_squared_error  # The functions necessary for the calculation of the RMSE error are imported.
print('Liberias cargadas')

#**************************************** END OF LIBRARIES TO USE *************************************

#***************************************** FUNCTIONS IMPLEMENTED ***********************************

# 1) eliminarFilasCorruptas
# This function is responsible for removing the rows that have negative values, -2700 and -1800 from the columns RH and AH
# of the training set "dataTrain".
# Delete rows that have negative values in the last column of dataTrain.

def eliminarFilasCorrputas(dataTrain):

    #A list is created to store negative column values.
    filas = []

    #Traverses the last column of dataTrain and checks if each element in each row is less than 0
    for i in range(dataTrain.shape[0]):
        if dataTrain.iloc[i,-1] < 0:
            #If less than 0 saves that negative value in rows
            filas.append(i)

    #An array is created to save negative values stored in the previously defined "filas" list.
    filas = np.array(filas)

    #The array "filas" is traversed
    for i in range(filas.shape[0]):
        #First dataTrain data becomes type "float32" in order to eliminate negative values.
        X = dataTrain.astype('float32')

        #Using the "drop" function of Python those negative values are removed from datatTrain.
        #The drop function is indicated:
        #1-labels=None since in this case the rows indicated by a label will not be removed.
        #2-axis=0 to indicate that you want to delete rows, not columns.
        #3-index=(rows[i]) to indicate which row you want to delete.
        dataTrain = X.drop(labels=None, axis = 0, index=(filas[i]))

    #A copy of the dataTrain training kit is returned with the rows, which had negative elements, removed.
    return dataTrain

print('Function: eliminarFilasCorrputas has been loaded')
#------------------------------------------------------------------------------------------------------------------

# 2) separarFilasNAN
# This function is responsible for separating the rows with NAN values in a column from the rows without NAN in that column.
# Used so that rows without NAN values can be used to estimate the NAN values of a column.

def separarFilasNAN(X_data, columna):

    # A list is created to store non-NAn values in a column.
    X_data_sinNAN = []

    # A list is created to store the NAN values of a column.
    X_data_conNAN = []

    # A list is created to store positions where there was a NAN value in a column.
    NAN_posicion = []

    # The training or test set is traversed
    for i in range(X_data.shape[0]):

        # Python’s "isnan" function checks whether the element is NAN or not
        # The function is entered:
        # 1- x=X_data, the array you want to check if a given column has NAN values or not.
        if np.isnan(X_data.iloc[i,columna]):
            # If the value of that row and column is NAN, the NAN value is stored in the previously defined list X_data_conNAN
            X_data_conNAN.append(X_data.iloc[i,:])

            # The position of the row in which that NAN value was located is stored in the previously defined list NAN_position
            NAN_posicion.append(i)

        else:
            #In the case that the value of that row and column is not NAN, the value is stored in the previously defined list X_data_sinNAN
            X_data_sinNAN.append(X_data.iloc[i,:])

    # This function returns:
    # 1- An array with the values stored in the X_data_sinNAN list.
    # 2- An array with the values stored in the X_data_conNAN list.
    # 3- An array with the values stored in the NAN_position list.
    return np.array(X_data_sinNAN), np.array(X_data_conNAN), np.array(NAN_posicion)

print('Function: separarFilasNAN has been loaded')
#------------------------------------------------------------------------------------------------------------------

# 3º) normalizacion
# This function is responsible for normalization the values of the training and test sets.
# This function is created because estimators are usually more effective when training and test data are normalized.

def normalizacion(X_cjto, mean_X = None, std_X = None):

    # If there is no calculated value of mean of X, calculate it
    if mean_X is None:
        mean_X = np.mean(X_cjto, axis=0)

    # If there is no calculated value of the standard deviation of X, calculate it
    if std_X is None:
        std_X = np.std(X_cjto, axis=0)

    # Data normalization is carried out
    X_norm = (X_cjto-mean_X)/std_X

    # This function returns:
    # 1-The data of the normalized training or test set.
    # 2-The calculated or used mean
    # 3-The calculated or used standard deviation value
    return X_norm, mean_X, std_X

print('Function: normalizacion has been loaded')
#--------------------------------------------------------------------------------------#

# 4º) estimadorGaussiano
# This function is responsible for performing a Gaussian process to train with the training data and to perform
# a Gaussian estimator to estimate the NAN values or the titanium values of the test set.
# The Ŝ values ​​of the X_test set are estimated when the training data X_train and S_train are passed to it.

def estimadorGaussiano(X_train_normalizado, X_test_normalizado, S_train,):

    # The hyperparameters to be used for the gaussian process are defined.
    sigmaf = np.std(S_train)
    sigma_eps = sigmaf / np.sqrt(1400)
    doslcuadrado = 50

    # The Euclidean distance between the X_train training data is calculated.
    # The Python "spatial.distance.cdist" function is used to calculate the distance
    # between each pair of the two collections of entries.
    # This function is passed to:
    # 1-The two input collections, in this case X_train_normalized and X_train_normalized.
    # 2-The metric to be used to heat the distance, in this case the euclidean
    distancia_train = spatial.distance.cdist(X_train_normalizado,X_train_normalizado,'euclidean')

    # The Euclidean distance between the X_train training data and the X_test data is also calculated
    # This is also done using the Python "spatial.distance.cdist" function, which
    # calculates the distance between each pair of the two collections of entries.
    # This function is passed:
    # 1-The two input collections, in this case X_test_normalized and X_train_normalized.
    # 2-The metric to be used to heat the distance, in this case the euclidean
    dist_test_train= spatial.distance.cdist(X_test_normalizado,X_train_normalizado,'euclidean')

    # In a Gaussian process you have to define your Kernel (covariance function)
    # An exponential quadratic function is choosed as a covariance function.
    K_train = (sigmaf**2)*np.exp(-np.power(distancia_train,1)/(doslcuadrado))
    K_test_train =(sigmaf**2)*np.exp(-np.power(dist_test_train,1)/(doslcuadrado))

    # Finally the estimate is made.
    # The following Python functions are used:
    # 1- The "ravel" function, which returns a 1-D matrix containing the input elements.
    # 2- The "dot" function, which performs the scalar product of two matrices.
    # 3- The function "np.linalg.inv", which calculates the inverse (multiplicative) of a matrix.
    # 4- The "np.eye" function, which returns a 2-D matrix with diagonal ones and zeros in the rest
    S_estimada = np.ravel(K_test_train.dot(np.linalg.inv(K_train + sigma_eps**2 * np.eye(K_train.shape[0]))).dot(S_train))

    #The estimate of either the NAN values or the titanium estimate is returned.
    return S_estimada

print('Function: estimadorGaussiano has been loaded')
#--------------------------------------------------------------------------------------#

# 5º) unirValoresEstiEnLosCjtos
# This function is responsible for joining the rows whose NAN values have been estimated and repositions them
# into the original training and test sets.
def unirValoresEstiEnLosCjtos(S_est, X_train, train_NAN_index, X_test,test_NAN_index,columna):

    # In S_train_estimado the values ​​of the S_est array are stored, ranging from position 0 to position equal to the number
    # of NAN indices that the training set "train" has.
    S_train_estimado = S_est[0:train_NAN_index.shape[0]]

    # In S_test_estimado the values ​​of the S_est array are stored, ranging from the position equal to the number of NAN indices
    # that the training set "train" has, to the number of NAN indices that the test set "test" has.
    S_test_estimado = S_est[train_NAN_index.shape[0]:(train_NAN_index.shape[0]+test_NAN_index.shape[0])]

    # Counter variables are initialized
    cont_train = 0
    cont_test = 0

    # The NAN indices that the "train" training set has are traversed.
    for i in range(train_NAN_index.shape[0]):

        # It is placed in each position where there was an NAN value the estimated value for that NAN.
        X_train.iloc[train_NAN_index[i],columna] = S_train_estimado[cont_train]
        # Increase the counter variable by one.
        cont_train+=1

    # The NAN indices that the "test" set has are traversed.
    for i in range(test_NAN_index.shape[0]):

        # It is placed in each position where there was an NAN value the estimated value for that NAN.
        X_test.iloc[test_NAN_index[i],columna] = S_test_estimado[cont_test]
        # Increase the counter variable by one.
        cont_test+=1

    # This function returns:
    # 1- The training set with the estimated NAN values.
    # 2- The test set with the estimated NAN values.
    return X_train, X_test

print('Function: unirValoresEstiEnLosCjtos has been loaded')
#--------------------------------------------------------------------------------------#

# 6º) estimated_error
# This function calculates the RMSE error of the estimator. It provides an idea of how good the method is and
# its chosen hyperparameters.
def estimated_error(Xn_train2, S_train2):

    # The training set is divided into:
    # 1- X_train
    # 2- S_train
    # 3- X_test
    # 4- S_real

    # In order to calculate the error RMSE between the real S "S_real" and the estimated S "S_est".
    # This way you can see how good or bad the implemented model is.

    # First, separate the data:
    X_train = Xn_train2.iloc[0:3500,:]
    S_train = S_train2.iloc[0:3500]

    X_test = Xn_train2.iloc[3500:-1,:]
    S_real = S_train2.iloc[3500:-1]

    # After that, the data is normalized.
    Xn_train, mean_X, std_X = normalizacion(X_train)
    Xn_test, mean_X, std_X = normalizacion(X_test, mean_X, std_X)

    # "S_est" values are estimated
    S_est = estimadorGaussiano(Xn_train,Xn_test,S_train)

    # The RMSE error is calculated.
    RMSE = mean_squared_error(S_real, S_est)**0.5

    # This function returns the RMSE error.
    return RMSE

print('Function: estimated_error has been loaded')
#--------------------------------------------------------------------------------------#

# 7º) export
# With this function the estimated "S_est" Titanium values are exported in .csv.
# The following format is set for the csv since the results that are uploaded to Kaggle require a specific format,
# which is the following.
def exportCsv(S_est):

    with open('Submissions.csv','w') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow(['Id','Prediction'])
        for i, x in enumerate(S_est):
            wr.writerow([i,x])

    print('The estimated data of PT08_S2_NMHC has been saved in ". csv" correctly.')

print('Function: export has been loaded')
#--------------------------------------------------------------------------------------#

#***************************************** END FUNCTIONS IMPLEMENTED ***********************************

#***************************************** IMPORT DATA FROM .CSV ****************************************

Data_train = pd.read_csv('.\input\data_train.csv', sep=',', dtype='float', header=0)
Data_test = pd.read_csv('.\input\data_test.csv', sep=',', dtype='float', header=0)

print('Import done correctly')

#************************************** END IMPORT DATA FROM .CSV ****************************************

#***************************************************** PREPROCESSING ************************************************

# 1º) The NMHC_GT column is removed from both the test and training sets as it contains too much lost data.
# To do this, first the data imported from the . csv of Data_train and Data_test becomes float32.
X = Data_train.astype('float32')
Y = Data_test.astype('float32')

# The Python drop function is then used to remove the NMHC_GT column from the training and test set.
# axis = 1 indicates that you want to remove the column.
data_train = X.drop('NMHC_GT',axis=1)
X_test = Y.drop('NMHC_GT', axis=1)

# 2º) The rows containing the values -2700 and -1800 in the columns RH and AH of the train set are deleted.
# To do this, we use the "eliminarFilasCorruptas" function explained above.
data_train = eliminarFilasCorrputas(data_train)

# 3º) The training X "X_train" is separated from the training set "S_train".
# To do this, first the data in the "data_train" training set becomes float32.
X = data_train.astype('float32')

# The integer column of PT08_S2_NMHC is stored in the S_train variable.
S_train = X.PT08_S2_NMHC

#Subsequently, using the Python "drop" function, the PT08_S2_NMHC column of the training set is removed
X_train = X.drop('PT08_S2_NMHC',axis=1)

# 4º) Data -900 becomes NAN (lost data).
X_train[X_train == -900] = np.nan
X_test[X_test == -900] = np.nan

#***************************************************** END PREPROCESSING ************************************************

#***************************************************** ALGORITHM/MODEL ************************************************

#***************************************************** ESTIMATION OF NAN VALUES **************************************

# The NAN values of the CO_GT, NOX_GT and NO2_GT columns are estimated for both the training and test sets.

# 1º) NAN values of NOX_GT are estimated

# First, the column CO_GT and NO2_GT are removed from both the training and test sets.

# All rows and all columns of X_train and X_test are taken.
X_train_1 = X_train.iloc[:,:]
X_test_1 = X_test.iloc[:,:]

# Python’s "drop" function removes the CO_GT and NO2_GT columns from both X_train and X_test.
X_train1_1 = X_train_1.drop('CO_GT',axis=1)
X_test1_1 = X_test_1.drop('CO_GT',axis=1)

X_train_NOX_GT = X_train1_1.drop('NO2_GT',axis=1)
X_test_NOX_GT = X_test1_1.drop('NO2_GT',axis=1)

# Secondly, using the "separarFilasNAN" function, the NAN rows are separated from the NAN-free rows
# of both the training and test sets.
# The function "separarFilasNAN" is passed the training set X_train_NOX_GT or the test set X_test_NOX_GT
# and passed a 1 since the column NOx_GT is column 1.
X_train_NOX_GT_clean, X_train_NOX_GT_NAN,  X_train_NOX_GT_NAN_index = separarFilasNAN(X_train_NOX_GT, 1)
X_test_NOX_GT_clean, X_test_NOX_GT_NAN, X_test_NOX_GT_NAN_index = separarFilasNAN(X_test_NOX_GT, 1)

# Third, once the rows with NAN are separated from those without NAN, in X_train_NOX_GT the values without NAN
# of both the training set and the test set are saved.
# The Python "concatenate" function is used
X_train_NOX_GT = np.concatenate((X_train_NOX_GT_clean, X_test_NOX_GT_clean),axis=0)

# In X_test_NOX_GT the NAN values of both the training and test sets are saved.
# This is done using the Python "concatenate" function
X_test_NOX_GT = np.concatenate((X_train_NOX_GT_NAN, X_test_NOX_GT_NAN),axis=0)

# Fourth, the separation of: X_train_NOX_GT, X_test_NOX_GT y S_train_NOX_GT.

# In S_train_NOX_GT the values in column 1 of X_train_NOX_GT are saved and are the training S values.
S_train_NOX_GT = X_train_NOX_GT[:,1]

# In X_train_NOX_GT the rest of the columns of X_train_NOX_GT are saved and will be the training X values.
X_train_NOX_GT = X_train_NOX_GT[:, (0,2,3,4,5,6,7)]

# In X_test_NOX_GT all the columns of X_test_NOX_GT are saved minus 1 which will be the test S values to be estimated.
X_test_NOX_GT = X_test_NOX_GT[:, (0,2,3,4,5,6,7)]

# Fifth, using the implemented function called "normalizacion", training and test sets are normalized.
X_train_NOX_GT_normalize, mean_X, std_X = normalizacion(X_train_NOX_GT)
X_test_NOX_GT_nomralize, mean_X, std_X = normalizacion(X_test_NOX_GT, mean_X, std_X)

# Sixth, the NAN values are estimated using a gaussian process.
S_NOX_GT_est = estimadorGaussiano(X_train_NOX_GT_normalize, X_test_NOX_GT_nomralize, S_train_NOX_GT)[:,np.newaxis]

# Finally, the rows are placed with the estimated NAN values in both the training sets and the original test sets.
X_train, X_test = unirValoresEstiEnLosCjtos(S_NOX_GT_est, X_train, X_train_NOX_GT_NAN_index, X_test, X_test_NOX_GT_NAN_index, 2)


# 2º) NAN values of NO2_GT are estimated

# First, the column CO_GT and NOx_GT are removed from both the training and test sets.

# All rows and all columns of X_train and X_test are taken.
X_train_2 = X_train.iloc[:,:]
X_test_2 = X_test.iloc[:,:]

# Python’s "drop" function removes the CO_GT and NO2_GT columns from both X_train and X_test.
X_train2_2 = X_train_2.drop('CO_GT',axis=1)
X_test2_2 = X_test_2.drop('CO_GT',axis=1)

X_train_NO2_GT = X_train2_2.drop('NOx_GT',axis=1)
X_test_NO2_GT = X_test2_2.drop('NOx_GT',axis=1)

# Second, the "separarFilasNAN" function separates the rows with NAN from the rows without NAN,
# both in the training and test sets.
# The "separarFilasNAN" function is passed the X_train_NO2_GT training set or the X_test_NO2_GT test set
# and passed a 2 as the column NO2_GT is column 2.
X_train_NO2_GT_clean, X_train_NO2_GT_NAN,  X_train_NAN_NO2_GT_index = separarFilasNAN(X_train_NO2_GT, 2)
X_test_NO2_GT_clean, X_test_NO2_GT_NAN, X_test_NAN_NO2_GT_index = separarFilasNAN(X_test_NO2_GT, 2)

# Third, once the rows with NAN are separated from those without NAN,
# in X_train_NO2_GT the values without NAN of both the training set and the test set are saved.
# The Python "concatenate" function is used
X_train_NO2_GT = np.concatenate((X_train_NO2_GT_clean, X_test_NO2_GT_clean),axis=0)

# In X_test_NO2_GT the NAN values of both the training and test sets are saved.
# This is done using the Python "concatenate" function
X_test_NO2_GT = np.concatenate((X_train_NO2_GT_NAN, X_test_NO2_GT_NAN),axis=0)

# Fourth, the separation of: X_train_NO2_GT, X_test_NO2_GT y S_train_NO2_GT.

# In S_train_NO2_GT the values in column 2 of X_train_NO2_GT are saved and are the training S values.
S_train_NO2_GT = X_train_NO2_GT[:,2]

# In X_train_NO2_GT the rest of the columns of X_train_NO2_GT are saved and will be the training X values.
X_train_NO2_GT = X_train_NO2_GT[:, (0,1,3,4,5,6,7)]

# In X_test_NO2_GT all the columns of X_test_NO2_GT are saved minus the 2 which will be the test S values to be estimated.
X_test_NO2_GT = X_test_NO2_GT[:, (0,1,3,4,5,6,7)]

# Fifth, using the "normalizacion" implemented function, training and test sets are normalized.
X_train_NO2_GT_normalize, mean_X, std_X = normalizacion(X_train_NO2_GT)
X_test_NO2_GT_normalize, mean_X, std_X = normalizacion(X_test_NO2_GT, mean_X, std_X)

# Sixth, the NAN values are estimated using a gaussian process.
S_NO2_GT_est = estimadorGaussiano(X_train_NO2_GT_normalize, X_test_NO2_GT_normalize, S_train_NO2_GT)[:,np.newaxis]

# Finally, the rows are placed with the estimated NAN values in both the training sets and the original test sets.
X_train, X_test = unirValoresEstiEnLosCjtos(S_NO2_GT_est, X_train, X_train_NAN_NO2_GT_index, X_test, X_test_NAN_NO2_GT_index, 4)

# 3º) The NAN values of CO_GT are estimated

# First, the column NOx_GT and NO2_GT are removed from both the training and test sets.

# All rows and all columns of X_train and X_test are taken.
X_train_3 = X_train.iloc[:,:]
X_test_3 = X_test.iloc[:,:]

# Using the Python "drop" function, the NOx_GT and NO2_GT columns of both X_train and X_test are removed.
X_train3_3 = X_train_3.drop('NO2_GT',axis=1)
X_test3_3 = X_test_3.drop('NO2_GT',axis=1)

X_train_CO_GT = X_train3_3.drop('NOx_GT',axis=1)
X_test_CO_GT = X_test3_3.drop('NOx_GT',axis=1)

# Secondly, using the "separarFilasNAN" function, the NAN rows are separated from the NAN-free rows
# of both the training and test sets.
# The function "separarFilasNAN" is passed the training set X_train_CO_GT or the test set X_test_CO_GT
# and passed a 0 since the column CO_GT is column 0.
X_train_CO_GT_clean, X_train_CO_GT_NAN, X_train_CO_GT_NAN_index = separarFilasNAN(X_train_CO_GT, 0)
X_test_CO_GT_clean, X_test_CO_GT_NAN, X_test_CO_GT_NAN_index = separarFilasNAN(X_test_CO_GT, 0)

# Third, once the rows with NAN are separated from those without NAN, in X_train_CO_GT the values without NAN
# of both the training set and the test set are saved.
# The Python "concatenate" function is used
X_train_CO_GT = np.concatenate((X_train_CO_GT_clean,X_test_CO_GT_clean),axis=0)

# In X_test_CO_GT the NAN values of both the interleaving and test sets are saved.
# This is done using the Python "concatenate" function
X_test_CO_GT = np.concatenate((X_train_CO_GT_NAN,X_test_CO_GT_NAN),axis=0)

# Fourth, the separation of: X_train_CO_GT, X_test_CO_GT y S_train_CO_GT.

# In S_train_CO_GT the values of column 0 of X_train_CO_GT are saved and are the training S values.
S_train_CO_GT = X_train_CO_GT[:,0][:,np.newaxis]

# In X_train_CO_GT the rest of the columns of X_train_CO_GT are saved and will be the training X values.
X_train_CO_GT = X_train_CO_GT[:,(1,2,3,4,5,6,7)]

# In X_test_CO_GT all the columns of X_test_CO_GT are saved minus 0 which will be the test S values to be estimated.
X_test_CO_GT = X_test_CO_GT[:,(1,2,3,4,5,6,7)]

# Fifth, using the "normalizacion" implemented function, training and test sets are normalized.
X_train_CO_GT_normalize, mean_X, std_X = normalizacion(X_train_CO_GT)
X_test_CO_GT_normalize, mean_X, std_X = normalizacion(X_test_CO_GT, mean_X, std_X)

# Sixth, the NAN values are estimated using a gaussian process.
S_CO_GT_est = estimadorGaussiano(X_train_CO_GT_normalize, X_test_CO_GT_normalize, S_train_CO_GT)[:,np.newaxis]

# Finally, the rows are placed with the estimated NAN values in both the training sets and the original test sets.
X_train, X_test = unirValoresEstiEnLosCjtos(S_CO_GT_est, X_train, X_train_CO_GT_NAN_index, X_test,X_test_CO_GT_NAN_index,0)

#*************************************************** END ESTIMATION OF NAN VALUES **************************************

#************************************************** ESTIMATION OF TITANIUM VALUES **************************************

# First, using the "normalizacion" function, training and test sets are normalized.
X_train_normalizado, mean_X, std_X = normalizacion(X_train)
X_test_normalizado, mean_X, std_X = normalizacion(X_test, mean_X, std_X)

# Secondly, we proceed to estimate the values of PT_08_S2_NMHC using a gaussian process.
S_PT_08_S2_NMHC_est = estimadorGaussiano(X_train_normalizado,X_test_normalizado,S_train)

# Finally, the RMSE error is calculated to check how good the implemented model is.
error = estimated_error(X_train, S_train)
print('The estimated error is: ', error)

#********************************************** END ESTIMATION OF TITANIUM VALUES **************************************

# CSV EXPORT WITH ESTIMATES

#csv file with titanium estimates is exported.
exportCsv(S_PT_08_S2_NMHC_est)
