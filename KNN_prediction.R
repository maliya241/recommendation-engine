#Setup from Recommendation_Engine.R
problem_data <- read.csv("problem_data.csv", header = TRUE)
train_submissions <- read.csv("train_submissions.csv", header = TRUE)
user_data <- read.csv("user_data.csv", header = TRUE)
sample_submission <- read.csv("sample_submission_SCGtj9F.csv", header = TRUE)
test_submissions <- read.csv("test_submissions_NeDLEvX.csv", header = TRUE)
test_submissions <- data.frame(attempts_range = rep(0, nrow(test_submissions)), test_submissions [,])
train_submissions$ID <- paste(train_submissions$user_id, train_submissions$problem_id)
data_combined <- rbind(test_submissions, train_submissions)
dc <- merge(x = data_combined, y = problem_data, by = "problem_id")
dc <- merge(x = dc, y = user_data, by = "user_id")
tspd <- merge(x = train_submissions, y = problem_data, by = "problem_id")
udtr <- merge(x = tspd, y = user_data, by = "user_id")
udtr$points.imp.mean <- ifelse(is.na(udtr$points), mean(udtr$points, na.rm=TRUE), udtr$points)

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

#for knn
library(neighbr)

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

library(caret) #For confusion matrix
library(e1071) #For confusion matrix

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


