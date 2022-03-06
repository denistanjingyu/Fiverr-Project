### Task: Build a Classification model to predict if the client will subscribe (yes/no) a term deposit (variable y).
### Interpret the model and assess its predictive performance. #####################################################
####################################################################################################################

# Import and prepare the Bank Marketing Dataset
bank = read.table(
  "C:\\Users\\user\\Desktop\\Project\\Part 3\\Dataset\\bank.csv",
  sep = ";",
  header = TRUE
)
summary(bank)
str(bank)
# Change y label from "yes" -> 1 and "no" -> 0
bank$y <- ifelse(bank$y == "yes", 1, 0)

### Logistic regression model
# Train/test split is a random process, and seed ensures the randomization works the same on different computers
set.seed(42)

library(caTools)
sampleSplit_1 <- sample.split(Y = bank$y, SplitRatio = 0.7)
trainSet_1 <- subset(x = bank, sampleSplit_1 == TRUE)
testSet_1 <- subset(x = bank, sampleSplit_1 == FALSE)

# Train 1 model using all features and predict results
# AIC: 1629.1
# Only 6 features are significant using a 5% significance threshold (loan, contact, month, duration, campaign, poutcome)
glm_model_1 <-
  glm(y ~ ., family = binomial(link = "logit"),  data = trainSet_1)
summary(glm_model_1)
# Predict a binary output using threshold of 0.5
pred_1 <- predict.glm(glm_model_1, testSet_1, type = "response")
pred_1[pred_1 > 0.5] <- 1
pred_1[pred_1 <= 0.5] <- 0

library(caret)
# Accuracy = 0.9004
acc_1 <-
  length(pred_1[pred_1 == testSet_1$y]) / length(pred_1)

# Repeat above steps for logistic regression model with the 6 significant variables
bank_sig_var <-
  bank[, c('loan',
           'contact',
           'month',
           'duration',
           'campaign',
           'poutcome',
           'y')]
set.seed(42)
sampleSplit_2 <-
  sample.split(Y = bank_sig_var$y, SplitRatio = 0.7)
trainSet_2 <-
  subset(x = bank_sig_var, sampleSplit_2 == TRUE)
testSet_2 <-
  subset(x = bank_sig_var, sampleSplit_2 == FALSE)
# AIC: 1625.3
glm_model_2 <-
  glm(y ~ ., family = binomial(link = "logit"),  data = trainSet_2)
summary(glm_model_2)
pred_2 <- predict(glm_model_2, testSet_2)
pred_2[pred_2 > 0.5] <- 1
pred_2[pred_2 <= 0.5] <- 0
# Accuracy = 0.8909 (Basically worse than using all features, significant features but not predictive)
acc_2 <-
  length(pred_2[pred_2 == testSet_2$y]) / length(pred_2)

# Try a decision tree model
# Use all features as decision tree has internal feature selection through split results
library(rpart)
tree_model_1 <- rpart(formula = y ~ .,
                      data = trainSet_1,
                      method = "class")
summary(tree_model_1)
pred_3 <- predict(tree_model_1, testSet_1)
pred_3 <- pred_3[, c('1')]
pred_3[pred_3 > 0.5] <- 1
pred_3[pred_3 <= 0.5] <- 0
# Accuracy = 0.9041 (Similar results to logistics regression)
acc_3 <-
  length(pred_3[pred_3 == testSet_1$y]) / length(pred_3)

# Retrieve optimal cp value based on cross-validated error
opt_index_1 <- which.min(tree_model_1$cptable[, "xerror"])
cp_opt_1 <- tree_model_1$cptable[opt_index_1, "CP"]
# Prune the model (to optimized cp value)
tree_model_2 <-
  prune(tree = tree_model_1, cp = cp_opt_1)
summary(tree_model_2)
pred_4 <- predict(tree_model_2, testSet_1)
pred_4 <- pred_4[, c('1')]
pred_4[pred_4 > 0.5] <- 1
pred_4[pred_4 <= 0.5] <- 0
# Accuracy = 0.9034 (Same as unpruned tree)
acc_4 <-
  length(pred_4[pred_4 == testSet_1$y]) / length(pred_4)

# Try XGBoost model
# XGBoost uses matrix data so that we need to convert our data into the xgb matrix type
library(xgboost)
library(caret)
xgb_train_x = data.matrix(trainSet_1[, -17])
xgb_train_y = trainSet_1[, 17]
xgb_test_x = data.matrix(testSet_1[, -17])
xgb_test_y = testSet_1[, 17]

xgb_train_1 = xgb.DMatrix(data = xgb_train_x, label = xgb_train_y)
xgb_test_1 = xgb.DMatrix(data = xgb_test_x, label = xgb_test_y)
xgb_model_1 = xgboost(
  data = xgb_train_1,
  max.depth = 2,
  nrounds = 50,
  objective = "binary:logistic"
)
# Acc = 0.9019 (Similar results)
pred_5 = predict(xgb_model_1, xgb_test_1)
pred_5[pred_5 > 0.5] <- 1
pred_5[pred_5 <= 0.5] <- 0
acc_5 <-
  length(pred_5[pred_5 == testSet_1$y]) / length(pred_5)