#from Recommendation_Engine.R
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

#data cleaning 
udtr$numeric_level_type <- as.numeric(udtr$level_type)-1 #maps each letter to a number (A - 1, B - 2, etc.)
udtr$numeric_rank <- ifelse(udtr$rank == "beginner", 1, ifelse(udtr$rank == "intermediate", 2, ifelse(udtr$rank == "advanced", 3, ifelse(udtr$rank == "expert", 4, 0)))) #maps each rank to a number (beginner - 1, intermediate - 2, advanced - 3, expert - 4)

# library(ggplot2)
# ggplot(data=udtr) + geom_point(mapping=aes(x=numeric_level_type, y=points.imp.mean))
# ggplot(data=udtr) + geom_point(mapping=aes(x=rank, y=points.imp.mean))
# ggplot(data=udtr) + geom_point(mapping=aes(x=rank, y=numeric_level_type))
#numeric_level_type and points.imp.mean produced to most interesting graph

#since the data set is too big, use a trimmed data set (1%)
#Using udtr
count(udtr)
#count = 155295
rows_count_udtr <- 155295
trimmed_number_of_rows_1 <- ceiling(rows_count_udtr*0.01)
trimmed_rows_1 <- sample(1:rows_count_udtr, trimmed_number_of_rows_1, replace=FALSE)
trimmed_udtr_1 <- subset(udtr[trimmed_rows_1, ])

#for knn
library(neighbr)

sampling_rate <- 0.8
testing_number_udtr_1 <- trimmed_number_of_rows_1 * (1-sampling_rate)

training_selected_rows_udtr_1 <- sample(1:trimmed_number_of_rows_1, sampling_rate*trimmed_number_of_rows_1, replace=FALSE)
training_set_udtr_1 <- subset(trimmed_udtr_1[training_selected_rows_udtr_1, ], select=c(numeric_level_type, points.imp.mean, attempts_range))
training_set_udtr_1$original_rowname <- as.numeric(rownames(training_set_udtr_1))
rownames(training_set_udtr_1) <- NULL

testing_selected_rows_udtr_1 <- setdiff(1:trimmed_number_of_rows_1, training_selected_rows_udtr_1)
testing_set_udtr_1 <- subset(trimmed_udtr_1[testing_selected_rows_udtr_1, ], select=c(numeric_level_type, points.imp.mean))
testing_set_udtr_1$original_rowname <- as.numeric(rownames(testing_set_udtr_1))
rownames(testing_set_udtr_1) <- NULL
predicted_testing_set_udtr_1 <- testing_set_udtr_1[ , ]
predicted_testing_set_udtr_1$attempts_range <- NA

true_labels <- trimmed_udtr_1$attempts_range[testing_selected_rows_udtr_1]

#Would be great to be able to do this part, but it takes too long just to do one knn
#for (k in 1:20) {
#   print(k)
#   predicted_labels <- knn(train_set=training_set_udtr_1, test_set=testing_set_udtr_1, k=k, categorical_target="attempts_range", comparison_measure="squared_euclidean", categorical_scoring_method="majority_vote")
#   num_incorrect_labels <- sum(predicted_labels != true_labels)
#   misclassification_rate <- num_incorrect_labels / testing_number_udtr_1
#   print(misclassification_rate)
# }

#need to select lowest k misclassification rate
selected_k = 1

predicted_attempts_range_1 <- knn(train_set=training_set_udtr_1, test_set=testing_set_udtr_1, k=selected_k, categorical_target="attempts_range", comparison_measure="squared_euclidean", categorical_scoring_method="majority_vote")
predicted_testing_set_udtr_1$attempts_range <- predicted_attempts_range_1$test_set_scores

write.csv(predicted_testing_set_udtr_1, file = "knn.csv", row.names = FALSE)