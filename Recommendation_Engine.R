###########################################################################################################################
# Recommendation Engine Challenge by Analytics Vidhya
# Trevor Judd, Maria McConkey, Kristen Patterson
# April 2020
###########################################################################################################################

#Libraries Needed
library(ggplot2) #For Creating Visualizations
library(stringr) #For String Manipulation
library(randomForest) #For Random Forest
library(caret) #For Cross Validation and Creating Confusion Matrix
library(doSNOW) #For Cross Validation
library(rpart) #For Cross Validation 
library(rpart.plot) #For Cross Validation
library(neighbr) #For K-Nearest Neighbors
library(e1071) #For Creating Confusion Matrix

###########################################################################################################################
# Data Setup
###########################################################################################################################

#Load raw data
problem_data <- read.csv("problem_data.csv", header = TRUE)
train_submissions <- read.csv("train_submissions.csv", header = TRUE)
user_data <- read.csv("user_data.csv", header = TRUE)
sample_submission <- read.csv("sample_submission_SCGtj9F.csv", header = TRUE)
test_submissions <- read.csv("test_submissions_NeDLEvX.csv", header = TRUE)

#Basically to start, we need to combine sample_submission with test_submissions and combine the resulting data frame with 
#train_submissions to form a data frame named data_combined

#Add a "attempts_range" variable to the test set to facilitate combining data sets
test_submissions <- data.frame(attempts_range = rep(0, nrow(test_submissions)), test_submissions [,])

#Add an "ID" variable to the train set to facilitate combining data sets
train_submissions$ID <- paste(train_submissions$user_id, train_submissions$problem_id)

#Now combine the test set and the train set using rbind
data_combined <- rbind(test_submissions, train_submissions)

#Voila! We now have a data frame named "data_combined" that contains all the relevant information from the following
#data frames: sample_submission, test_submissions, and train_submissions
#Now we need to continue merging data_combined with problem_data and user_data

dc <- merge(x = data_combined, y = problem_data, by = "problem_id")
#Now we have a data frame, "dc", which combines the data_combined and problem_data data frames

dc <- merge(x = dc, y = user_data, by = "user_id")
#Now dc contains all the data we were given to solve the problem of predicting "attempts_range"

#R data types
str(dc)

#Take a look at gross attempts
table(dc$attempts_range)
#Note 66,555 entries with No Attempts (0).  This is the test set portion of the combined data set
#Metadata says:
#attempts_range                         Num. of attempts
#  1                                         1-1
#  2                                         2-3
#  3                                         4-5
#  4                                         6-7
#  5                                         8-9
#  6                                         >=10

###########################################################################################################################
# Exploratory Data Analysis
###########################################################################################################################

#Hypothesis: the higher the difficulty level (level_type) the fewer the number of attempts
ggplot(dc, aes(x = level_type, fill = factor(attempts_range))) +
  stat_count(width = 0.5) +
  xlab("Level Type") +
  ylab("Total Count") +
  labs(fill = "attempts_range")

#This may seem like a "duh" observation, but as you can see from the bar graph, people tend to submit solutions to "easy"
#problems more than they submit solutions to "difficult" problems, but this observation does support our hypothesis

#Are there any duplicated rows of information?  A user may have worked multiple problems and will therefore have
#multiple entries with the same user_id but different problem information
dc[duplicated(dc) | duplicated(dc, fromLast=TRUE), ]

#We have all unique rows! Therefore, all the data is unique. We can proceed.

ggplot(udtr, aes(x = rank, fill = factor(attempts_range))) + 
  stat_count(width = 0.5, color="black") + 
  xlab("rank") + 
  ylab("Total Count") + 
  labs(fill = "attempts_range")
#There is some predictive power in the user's rank (beginner, intermediate, advanced, expert)?

#Is there any correlation with other variables?
tags <- dc[which(str_detect(dc$tags, "brute force")),]
tags[1:5,]

#Hypothesis: there is a correlation between user rank and problem tag
tags2 <- dc[which(str_detect(dc$tags, "binary search")),]
tags2[1:5,]

tags3 <- dc[which(str_detect(dc$tags, "two pointers")),]
tags3[1:5,]

#Note: "brute force" and "binary search" problems tend to be done by "advanced" users
#Some "two pointers" problems are done by intermediate-level users
#Note lines 24 and 48; two problems classified level C have different point values
#Note line 14 where level_type is E and points are 2500, but the problem has fewer tags and the one belopw it
#Conclusion: the number of tags and the tags themselves probably can't be used to make reliable predictions
#but level_type, points, and rank might be correlated
#Also, points cannot be used to fill in NA values for level-type and vice versa

#We should expand upon the relationship between level_type, points, and rank. 

#Create a new data frame named "udtr" which combines user_data and train_submissions (we need to do this because 
#the train set does not have any zeros in its attempts_range column)
tspd <- merge(x = train_submissions, y = problem_data, by = "problem_id")
udtr <- merge(x = tspd, y = user_data, by = "user_id")

#We are interested in whether a user's rank and a problem's level_type affects the user's attempt_range.
ggplot(udtr, aes(x = level_type, fill = factor(attempts_range))) +
  stat_count(width = 0.5) +
  facet_wrap(~rank) +
  ggtitle("rank") +
  xlab("level_type") +
  ylab("Total Count") +
  labs(fill = "attempts_range")

#We have a peculiar stair-step pattern with Beginners and Intermediates but Advanced and Experts overall make fewer
#attempts, but the drop off is the same at all four levels.  The data indicates that, regardless of rank, users 
#attempt more easier problems than harder ones, and Advanced and Expert users make fewer attempts overall.

#What is the distribution of user ranks across the dataset?
table(udtr$rank)

#Most users are Intermediates. The smallest rank is Expert.  Most of our prediction data is going to come from Beginners
#and Intermediates

#Visualize the 3-D relationship between points, rank, and attempts_range
ggplot(udtr, aes(x = points, fill = factor(attempts_range))) +
  geom_histogram(binwidth = 100) +
  facet_wrap(~rank) +
  ggtitle("rank") +
  xlab("points") +
  ylab("Total Count") +
  labs(fill = "attempts_range")

#There does seem to be some predictive power in the points variable. Our graphs have the same general shape as our plot
#linking level_type and rank.  I think this is a good indicator that our best variables to use for modeling are
#level_type and rank.

#Note the number of NA vaules.  In order to proceed, we must do one of two things: replace the NAs with mean
#or median values, or you can train a model (linear regression, K-clustering, etc...) to fill in the missing 
#values using data variables that are complete (and that method that works very well)

###########################################################################################################################
# Random Forest
###########################################################################################################################

#Train a Random Forest with the default parameters level_type & rank
rf.train.1 <- udtr[c("level_type", "rank")]
rf.label <- as.factor(udtr$attempts_range)
set.seed(1234)
rf.1 <- randomForest(x = rf.train.1, y = rf.label, importance = TRUE, ntree = 1000)
rf.1
varImpPlot(rf.1)

#As can be seen from the plot, the level_type is the more accurate predictor

#Create a new column in udtr that fills NA values in the points column with the column mean
udtr$points.imp.mean <- ifelse(is.na(udtr$points), mean(udtr$points, na.rm=TRUE), udtr$points)

#Train a Random Forest using level_type, rank, and points
rf.train.2 <- udtr[c("level_type","rank","points.imp.mean")]
set.seed(1234)
rf.2 <- randomForest(x = rf.train.2, y = rf.label, importance = TRUE, ntree = 1000)
rf.2
varImpPlot(rf.2)

#When used in combination, points, level type, and rank produces a marginally better predictive model

#Train a Random Forest with the default parameters using level_type, rank, points, and rating
rf.train.3 <- udtr[c("level_type","rank","points.imp.mean","rating")]
set.seed(1234)
rf.3 <- randomForest(x = rf.train.3, y = rf.label, importance = TRUE, ntree = 1000)
rf.3
varImpPlot(rf.3)

#We get a similar error rate.  Let's try max rating instead.

rf.train.4 <- udtr[c("level_type","rank","points.imp.mean","max_rating")]
set.seed(1234)
rf.4 <- randomForest(x = rf.train.4, y = rf.label, importance = TRUE, ntree = 1000)
rf.4
varImpPlot(rf.4)
#Slight drop in accuracy.

#Train a Random Forest with the default parameters using level_type, rank, points, and contribution
rf.train.5 <- udtr[c("level_type","rank","points.imp.mean","contribution")]
set.seed(1234)
rf.5 <- randomForest(x = rf.train.5, y = rf.label, importance = TRUE, ntree = 1000)
rf.5
varImpPlot(rf.5)
#Equally accurate to rf.2

#Train a Random Forest with the default parameters using level_type, rank, points, and submission count
rf.train.7 <- udtr[c("level_type","rank","points.imp.mean","submission_count")]
set.seed(1234)
rf.7 <- randomForest(x = rf.train.7, y = rf.label, importance = TRUE, ntree = 1000)
rf.7
varImpPlot(rf.7)
#This actually yields the lowest error rate so far! (46.5%)

#Train a Random Forest with the default parameters using level_type, rank, points, submission count, and max rating
rf.train.8 <- udtr[c("level_type","rank","points.imp.mean","submission_count","max_rating")]
set.seed(1234)
rf.8 <- randomForest(x = rf.train.8, y = rf.label, importance = TRUE, ntree = 1000)
rf.8
varImpPlot(rf.8)
#Error jumps up again.  Looks like rf.7 is our most accurate model (such as it is).

#Before we jump into features engineering, we need to establish a methodology for estimating our error rate on the 
#test set (i.e. unseen data). This is critical because without this we are more likely to overfit.  

#Create a new column in dc that fills NA values in the points column with the column mean
dc$points.imp.mean <- ifelse(is.na(dc$points), mean(dc$points, na.rm=TRUE), dc$points)

#Create a new data frame from dc 
dc_test_data <- subset(dc, attempts_range==0)

#Subset our test records and features
test.submit.df <- dc_test_data[c("level_type","rank","points.imp.mean","submission_count")]

#Make predictions
rf.7.predict <- predict(rf.7, test.submit.df)
table(rf.7.predict)

#Write out a CSV file for submission to Data Hack
submit.df <- data.frame(ID = dc_test_data$ID, attempts_range = rf.7.predict)
write.csv(submit.df, file = "RF_SUB_04162020_1.csv", row.names = FALSE)

#We got back a score of 0.403 using this model's predictions.  Considering the highest score is 0.51, this isn't 
#too bad for a first attempt!

rf.train.7.1000 <- subset(rf.train.7[1:1000,])
rf.train.7.1000 <- subset(rf.train.7[1:2])
rf.label.1000 <- rf.label[1:1000]
rf.train.7.1000$level_type <- sub("^$", "B", rf.train.7.1000$level_type)

###########################################################################################################################
# Cross Validation 
###########################################################################################################################

#Let's look into cross-validation using the caret package to see if we can get 
#more accurate estimates

#3-fold cross-validation

set.seed(37596)
cv.3.folds <- createMultiFolds(rf.label.1000, k = 3, times = 10)

ctrl.3 <- trainControl(method = "repeatedcv", number = 3, repeats = 10, index = cv.3.folds)

cl <- makeCluster(6, type = "SOCK")
registerDoSNOW(cl)

set.seed(94622)
rf.7.cv.3 <- train(x = rf.train.7.1000, y = rf.label.1000, method = "rf", tuneLength = 3, ntree = 1000, trControl = ctrl.3)

#Shutdown cluster
stopCluster(cl)

#check out results
rf.5.cv.3

rpart.cv <- function(seed, training, labels, ctrl) {
  cl <- makeCluster(6, type = "SOCK")
  registerDoSNOW(cl)
  
  set.seed(seed)
  #Leverage formula interface for training
  rpart.cv <- train(x = training, y = labels, method = "rpart", tuneLength = 30, trControl = ctrl)
  
  #Shutdown cluster
  stopCluster(cl)
  
  return (rpart.cv)
}

features <- c("level_type","rank","points.imp.mean","submission_count")
rpart.train.1 <- udtr[1:1000, features]

#Run CV and check out results
rpart.1.cv.1 <- rpart.cv(94622, rpart.train.1, rf.label, ctrl.3)
rpart.1.cv.1

#Inconclusive results due not being able to execute the cross validation with current hardware.

###########################################################################################################################
# K-Nearest Neighbors
###########################################################################################################################

#Transforming data to numeric because K-NN requires numeric data 
udtr$numeric_level_type <- as.numeric(udtr$level_type)-1 #maps each letter to a number (A - 1, B - 2, etc.)
udtr$numeric_rank <- ifelse(udtr$rank == "beginner", 1, ifelse(udtr$rank == "intermediate", 2, ifelse(udtr$rank == "advanced", 3, ifelse(udtr$rank == "expert", 4, 0)))) #maps each rank to a number (beginner - 1, intermediate - 2, advanced - 3, expert - 4)

#Seeing how the different variables interact with each other
library(ggplot2)
ggplot(data=udtr) + geom_point(mapping=aes(x=numeric_level_type, y=points.imp.mean))
ggplot(data=udtr) + geom_point(mapping=aes(x=rank, y=points.imp.mean))
ggplot(data=udtr) + geom_point(mapping=aes(x=rank, y=numeric_level_type))
#numeric_level_type and points.imp.mean produced to most interesting graph

#Since the data set (udtr) is too big, use a trimmed data set (1%)
#Already tried trimming to 50%, 10%, 5% and still took too long
rows_count_udtr <- nrow(udtr) #Getting how many rows in udtr
trimmed_number_of_rows_1 <- ceiling(rows_count_udtr*0.01) #Figuring out how many rows is 1%
trimmed_rows_1 <- sample(1:rows_count_udtr, trimmed_number_of_rows_1, replace=FALSE) #Selecting rows
trimmed_udtr_1 <- subset(udtr[trimmed_rows_1, ]) #Creating trimmed subset

sampling_rate <- 0.8 #Selected training percent 
testing_number_udtr_1 <- trimmed_number_of_rows_1 * (1-sampling_rate) #Getting the number of the testing set rows from the 1%

training_selected_rows_udtr_1 <- sample(1:trimmed_number_of_rows_1, sampling_rate*trimmed_number_of_rows_1, replace=FALSE) #Selecting rows from the trimmed subset
training_set_udtr_1 <- subset(trimmed_udtr_1[training_selected_rows_udtr_1, ], select=c(numeric_level_type, points.imp.mean, attempts_range)) #Creating training subset with numeric_level_type, points.imp.mean, attempts_range
training_set_udtr_1$original_rowname <- as.numeric(rownames(training_set_udtr_1)) #Adding column to include the original index
rownames(training_set_udtr_1) <- NULL #Reindexing the subset to make it match later

testing_selected_rows_udtr_1 <- setdiff(1:trimmed_number_of_rows_1, training_selected_rows_udtr_1) #Selecting the rest of the unselected 1%
testing_set_udtr_1 <- subset(trimmed_udtr_1[testing_selected_rows_udtr_1, ], select=c(numeric_level_type, points.imp.mean)) #Creating the testing subset without attempts_range
testing_set_udtr_1$original_rowname <- as.numeric(rownames(testing_set_udtr_1)) #Adding column to include original index
rownames(testing_set_udtr_1) <- NULL #Reindexing the subset 
predicted_testing_set_udtr_1 <- testing_set_udtr_1[ , ] #Creating a copy of the testing subset
predicted_testing_set_udtr_1$attempts_range <- NA #Including attempts_range with NA values, this is where predicted values will be place

true_labels <- subset(trimmed_udtr_1[testing_selected_rows_udtr_1, ], select=c(numeric_level_type, points.imp.mean, attempts_range)) #Making a subset with the testing rows to include attempts_range so we know what the original labels were
true_labels$original_rowname <- as.numeric(rownames(true_labels)) #Adding column to include original index
rownames(true_labels) <- NULL #Reindexing the subset

#Select the best k based on the lowest misclassification rate
#Would be great to be able to do this part, but it takes too long just to do one knn
#for (k in 1:20) {
#   print(k)
#   predicted_labels <- knn(train_set=training_set_udtr_1, test_set=testing_set_udtr_1, k=k, categorical_target="attempts_range", comparison_measure="squared_euclidean", categorical_scoring_method="majority_vote")
#   num_incorrect_labels <- sum(predicted_labels != true_labels)
#   misclassification_rate <- num_incorrect_labels / testing_number_udtr_1
#   print(misclassification_rate)
# }

#Need to select lowest k misclassification rate
selected_k = 1

#Execute K-NN 
predicted_attempts_range_1 <- knn(train_set=training_set_udtr_1, test_set=testing_set_udtr_1, k=selected_k, categorical_target="attempts_range", comparison_measure="squared_euclidean", categorical_scoring_method="majority_vote")

#Storing results and preparing to create the confusion matrix
predicted_testing_set_udtr_1$attempts_range <- predicted_attempts_range_1$test_set_scores #Placing K-NN prediction results into the prediction subset
prediction_attempts_range <- predicted_testing_set_udtr_1$attempts_range #Separating predicted attempts_range into its own dataframe
prediction_attempts_range$categorical_target <- as.factor(prediction_attempts_range$categorical_target) #Make predicted attempts_range a factor
predict_attempts_range <- prediction_attempts_range$categorical_target #Making it into a list
true_labels$attempts_range <- as.factor(true_labels$attempts_range) #Making true labels attempts_range into factor
correct_attempts_range <- true_labels$attempts_range #Making it into a list

#Create confusion matrix
confusion_matrix <- confusionMatrix(predict_attempts_range, correct_attempts_range, mode="prec_recall")

#For running multiple times; Gave a letter to separate each iteration
predict_attempts_range_ <- predict_attempts_range
correct_attempts_range_ <- correct_attempts_range

#Ran Selecting the 1% to executing K-NN and storing the results five times 
#Following code is for visualization purposes
f1 <- c(0.5855, 0.27778, 0.14815, NaN, NA, NaN, 0.5390, 0.3265, 0.056338, NaN, NaN, NaN, 0.5153, 0.3585, 0.16129, 0.066667, NaN, NaN, 0.5401, 0.30000, 0.071429, 0.071429, NaN, NaN, 0.5610, 0.3168, 0.19608, 0.095238, NaN, NaN) #Had to hard code the values in
class <- c(1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6) #Had to hard code the values in
#iteration <- c("a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "c", "c", "c", "c", "c", "c", "d", "d", "d", "d", "d", "d", "e", "e", "e", "e", "e", "e") #Had to hard code the values in
iteration <- c(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5) #Had to hard code the values in
f1_scores <- data.frame("f1_score" = 1:30, "class" = 1:30, "iteration" = 1:30)
f1_scores$f1_score <- f1
f1_scores$class <- as.factor(class)
f1_scores$iteration <- as.factor(iteration)
ggplot(f1_scores, aes(x=as.factor(class), y=f1_score, group=iteration)) + geom_line(size=1.5, aes(linetype=iteration, color=iteration)) + geom_point(size=3, aes(color=iteration)) + xlab("attempts_range") + ylab("f1_score") + scale_colour_manual(values = c("#9CF3F9", "#6CE2EF", "#0A97CF", "#045DAF", "#063677", "#042B6C")) 
