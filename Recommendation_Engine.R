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

#Exploratory data analysis
library(ggplot2)

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

#There is some predictive power in the user's rank (beginner, intermediate, advanced, expert)?
library(stringr)

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

library(randomForest)

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

#Let's look into cross-validation using the caret package to see if we can get 
#more accurate estimates
library(caret)
library(doSNOW)

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

#Install and load packages
library(rpart)
library(rpart.plot)

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

#I don't know what the problem is.






















