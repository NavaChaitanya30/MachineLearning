# Our aim is to predict the type of Species of a flower Based on its attributes.
# like .. if the Sepal Length is 5.1 cm ,Sepal Width is 3.5 cm,petal length is 1.4 cm and Petal width is 0.2 cm
# then the flower belongs to a species called .. 'Iris-setosa'.(look at the first row of the dataframe down below)

#importing all the required modules
from sklearn.linear_model import LinearRegression # it is the Linear Regression learning model.
model=LinearRegression()  #creating an object for LR model which is further used to train with the data.

# since Linear Regression model deals with the numerical Values .. which means it takes the input in the form of 
# numerial values and produce the output in numeric format..But in our dataset the input values are numeric (no offence)
# where as the output is a string (i.e., "Iris-setosa" or "Iris-virginica" or "Iris-versicolor")..
#which is not a numeric data type  .. so we need to convert this values into numeric data type..
# let's say we replaced  "Iris-setosa" with 0(zero),"Iris-virginica" with 1,and "Iris-versicolor" with 2..
# then we'll end up having a numeric value in output lable too.
# now it is easy for a linear regression model to intake numeric values and produce a numeric value as an output...(i.e., 0,1 or 2)

# the following LableEncoder module helps us in converting the catogorical values (i.e, strings) into an unic integer value.
from sklearn.preprocessing import LabelEncoder # this is used to encode the catogorical label into numerical value.

import pandas as pd

#loading dataset from the google drive
# DataFrame = pd.read_csv(location)
df1=pd.read_csv("/content/drive/MyDrive/ML-lab/datasets/Iris.csv")
df1.head()

#upto now we just loaded the csv file from the specified locatin. In our scenario, we did it from google drive.
# checking the output label (i.e, Species),it has the categorical values(i.e, string ---which is not a numeric value)

df1['Species'].head()

# this code is used to replace those strings with an unique number.
import numpy as np
Y=[]  # we take an empty list.We can append values later.
for i in df1['Species']:  # for each and every species name in the dataframe(df1['species])
  if i=='Iris-setosa':  # check wheather it matches with the Species "'Iris-setosa'"
    Y.append(0) # then we append '0' into the list 
  elif i=='Iris-versicolor':  # check wheather it matches with the Species 'Iris-versicolor'
    Y.append(1) # then we append '1' into the list 
  elif i=='Iris-virginica':# check wheather it matches with the Species 'Iris-virginica'
    Y.append(2) # then we append '2' into the list 
Y=np.array(Y) # covert the list into array using numpy.
Y=Y.reshape(-1,1) # reshaping array so that it holds only one column related data .. this is because we are assigning only one column as output lable
print("previous dataframe of species column. it will print the strings\ndf1['species']")
print(df1['Species'].head())# check the previous dataframe of species column. it will print the strings 

df1.drop('Species',axis=1,inplace=True)# now we remove those strings from the data frame.
#
#    | df1['species']  |     Y
#------------------------------
# 0  | 'Iris-setosa'   |-->  0
# 70 |'Iris-versicolor'|-->  1
# 135|'Iris-virginica' |-->  2
#
df1['Species']=Y # assigning numerical values into the dataframe column[species]
print("modified DataFrame with numerical valuses \ndf1['species']")
print(df1['Species'].head())

# if we use python in-built methods..
# The above task can be done in 3 lines of code..
le=LabelEncoder()# created an object for Label encoder method
Y1=le.fit_transform( df1['Species'])# passing the column of the dataframe for which we want to encode the numerical values instead of strings.
Y1

# So far we are successfull replacing the strings with numerical values so that our LR model can train on it.
# sometimes there will be a case where the data is missing in a particular column and row.
# we can check wheather there are any null values in the dataframe using following code.
df1.isnull().sum()

# that  prints out the count of the missing data in each colums.We are lucky if each column contains 0 null values.
# if not we can replace those null values using mean , mode or median of each and every column.
# these all techniques are called or comes under a method called Data-Preprocessing.

# Ok.
# The actual need of Data-Preprocessing is to avoid the model from learning bad patterns of data.
# Now we are successfull in designing or making a clean dataset.. without null values .. without wrong formats of data.
# Now the data set is ready .. So our model can train on this Dataset .
# Before the Model train on our dataset, We have an important task i.e, we need to divide the data set into Two parts.
# First part contains the input data,and the second contains Output data ,Mathematically 1 is X and 2nd is Y.

# we are now aware of the output i.e 0(Iris-setosa), or 1('Iris-versicolor') or 2('Iris-virginica')...
# Output is.. Species ---->Y
print("OUTPUT LABEL\n\n  Species")
Y1=df1['Species']
print(Y1.head())
# then the input will obviously the remaining columns..i.e, SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm
X1= df1.drop(columns=['Id','Species'])# we are deleting the output label from the input set (i.e., Species) .sinse Id is useless we remove it too.
print("\nINPUT LABEL\n")
print(X1.head())

# Since We would like to test our model performance after it learns from the dataset.
# We need testing samples..
# in-oredr to get tose samples we further divide our input dataset into two parts in the ratio 70% and 30%...
# since we are taking the input testing samples from our data set now we have an advantage that our dataset contains the corresponding
#output in the output set. (i.e, Y set) so we divide the Y set into 2 parts in the same ratio as X set.
# afte doing this .. We end-up having 4 datasets or data cubes..1 st) 70% of X set for training 2nd) 30% X set for testing.
# 3 rd) 70% of Y set for Training ..4 th) 30% of Y set for testing.
#
#   X_train     |   Y_train     |   X_test    |   Y_test    |
#   
#     70%       |     70%       |     30%     |     30%     |
#


# ush...
# in order to split the Dataset into 4 subset with specified percentage(%) of testing samples..
# python skykit learn provides a module called train_test_split which is availabla in sklearn.model_selection package.

# importing ..
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y1,test_size=0.3)# this line of code retuns 4 tuples that we store them in X_train, X_test, Y_train, Y_test
# we can specify the test size .. we gave 30% for test size, because it is the best ratio for avoiding over-fitting and under-fitting problems.
# the above function or method choses randomly 70% data for training and the ramaining for testig.

# the above function returns the 4 subsets 
#
#   X_train     |   Y_train     |   X_test    |   Y_test    |
#               |               |             |             |
#     70%       |     70%       |     30%     |     30%     |
#

print(X_train.shape[0]," rows are selected randomly for training")
X_train.head()

print("Remaining ",X_test.shape[0],"are for testing")
X_test.head()

print(Y_train.shape[0]," rows are selected randomly for training")
Y_train

print("Remaining ",Y_test.shape[0],"are for testing")
Y_test

#Happy..
# So far we are done doing everything so that model can train on the data and predict with the help of data samples.
#Now it's time to roll our sleves and jump for the next step. That is .. training the model with the X and Y training sets.

model.fit(X_train,Y_train) # fit() method is the method that plays a crucial role in training the ML model.


#yeah...
# Our model is ready now. now we can test this model using those(X_test) testing samples.
# 
Y_pred=model.predict(X_test)# testing the model with the 30% data from the actual dataframe.
#for every X value .. Our model predicts a Y value.So those values are again stored in Y_pred set.

# 
# testing samples|model predicted   ||  Actual values
#  |   X_test    |   Y_pred         ||   Y_test    |
#  |      1      |     0            ||     0       |
#  |      2      |     0            ||     1       |
#  |      3      |     2            ||     2       |

#Actual values
Y_test

# model predicted values
Y_pred

# we can clearly observe the diffrence between Actual and predicted values.
#so we should calculate accuracy of the trained model by few  performence metrix.

#mean absolute error calculation(MAE)
#mean square error(MSE)
#Root Mean Square Error(RMSE)
#root(MSE)
#R^2 
#SST
#SSE

#mean square error(MSE)
sum=0
j=0
for i in Y_test:
  sum=sum+(i-Y_pred[j])**2
  
  j=j+1
#MSE
MSE=sum/len(Y_test)
MSE

# over all mean of Y_test set
ybar=Y_test.mean()

#SST
sum=0

for i in Y_test:
  sum=sum+(i-ybar)**2
SST=sum/len(Y_test)
SST

#SSE
SSE=MSE
SSE

R2=(SST-SSE)/SST
print("Accuracy of our model is:",R2*100)

# To avoid this much of code. we can simply use python inbuilt methods for claculating the score.
score=model.score(X_test,Y_test)
print("Accuracy:",score*100)

# visualisation

import matplotlib.pyplot as plt
plt.plot(Y_test,Y_pred)
plt.scatter(x=Y_test,y=Y_pred,color="red")
plt.show()

#done.
# our model is trained .. Now we can save this model in our drive.
import joblib
joblib.dump(model,'/content/drive/MyDrive/ML-lab/mymodels/iris_LR.joblib')
# the advantage of saving the trained model is...
# we can load this model and use it for predictions. without again doing all this preprpcesing and training.

# loading model..
mymodel = joblib.load('/content/drive/MyDrive/ML-lab/mymodels/iris_LR.joblib')

species = mymodel.predict([[5.1	,3.5	,1.4,	0.2]]) # passing  a single sample to model.

print("Model predicted value:",species[0],"\n\n")


# if the value is near to 0 which means the species would be  'Iris-setosa'
#if the value is near to  1 which means the species would be  'Iris-versicolor' 
#if the value is near to  2 which means the species would be  'Iris-virginica'