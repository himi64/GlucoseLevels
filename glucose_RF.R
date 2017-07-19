###################################################################################
# file: glucose_RF>R
# author: Himanshu

###################################################################################
### PART 1: RANDOM FOREST REGRESSION WITH ALL VARIABLES

# Importing the dataset
diabetes = read.csv('diabetes2.csv')

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(diabetes, SplitRatio = 0.75)
training_set = subset(diabetes, split == TRUE)
test_set = subset(diabetes, split == FALSE)

# Fitting Random Forest Regression to the dataset
install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = diabetes[1:8],
                         y = diabetes$plasma.glucose,
                         ntree = 500)

# predicting results on the test set
y_test_pred = predict(regressor, test_set)

y_test_pred

# Predicting a new result with Random Forest Regression
y_pred = predict(regressor, data.frame(pregnant = 2,
                                       diastolic = 80,
                                       tricepsfold = 10,
                                       X2hrinsulin = 10,
                                       bmi = 35,
                                       dbtspedigree = 0.9,
                                       age = 50,
                                       diabetes = 1))

y_pred

# confusion matrix of test set results
#install.packages('caret')
#library(caret)
#cm = confusionMatrix(y_test_pred, test_set)

# Visualising the Random Forest Regression results using clarke error grid
# install.packages('ega')
library(ega) # for plotting clarke error grid

plotClarkeGrid(y_test_pred, test_set$plasma.glucose)

#####################################################################################
### PART 2: RANDOM FOREST REGRESSION WITHOUT TRICEPS, PREGNANT, DBTSPEDIGREE, INSULIN 

# read data
diabetes2 = read.csv('diabetes4.csv')

# split into training and testing
set.seed(123)
split = sample.split(diabetes2, SplitRatio = 0.75)
training_set2 = subset(diabetes2, split == TRUE)
test_set2 = subset(diabetes2, split == FALSE)

# Fitting Random Forest Regression to the dataset
regressor2 = randomForest(x = diabetes[1:4],
                          y = diabetes$plasma.glucose,
                          ntree = 500)

# predict results on the test set
predict(regressor2, test_set2)
