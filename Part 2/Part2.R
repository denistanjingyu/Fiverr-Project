### Task: Build a regression model for the variable  G3 (final grade) without using the variables G1 and G2. ###
### Interpret the model and assess its predictive performance. #################################################
################################################################################################################

# Import and prepare the student performance dataset
school1 = read.table(
  "C:\\Users\\user\\Desktop\\Project\\Part 2\\Dataset\\student-mat.csv",
  sep = ";",
  header = TRUE
)
school2 = read.table(
  "C:\\Users\\user\\Desktop\\Project\\Part 2\\Dataset\\student-por.csv",
  sep = ";",
  header = TRUE
)
# schools = merge(school1,
#                 school2,
#                 c("school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
#                   "reason", "nursery", "internet"))

# Pre-processing
# Remove G1 and G2 from both datasets
school1 = subset(school1, select = -c(G1, G2))
school2 = subset(school2, select = -c(G1, G2))

# Summary of both datasets (descriptive statistics)
summary(school1)
summary(school2)

### Linear regression model for mathematics dataset
# Train/test split is a random process, and seed ensures the randomization works the same on different computers
set.seed(42)
# Use default 70/30 train/test split
library(caTools)
sampleSplit_mat_1 <- sample.split(Y = school1$G3, SplitRatio = 0.7)
trainSet_mat_1 <- subset(x = school1, sampleSplit_mat_1 == TRUE)
testSet_mat_1 <- subset(x = school1, sampleSplit_mat_1 == FALSE)

# Train 1 model using all features and predict results
# Adjusted R-squared: 0.1375 is low/Model p-value: 0.000312
# Only 2 features are significant using a 5% significance threshold (failures, romanticyes)
model_mat_1 <- lm(G3 ~ . , data = trainSet_mat_1)
summary(model_mat_1)
preds_mat_1 <- predict(model_mat_1, testSet_mat_1)
# RMSE = 3.9742 (Quite large error as G3 range is 0-20)
library(Metrics)
rmse_mat_1 <-
  rmse(actual = testSet_mat_1$G3, predicted = preds_mat_1)

# Repeat above steps for linear regression model with the 2 significant variables
school1_mat_sig_var <-
  school1[, c('failures', 'romantic', 'G3')]
sampleSplit_mat_2 <-
  sample.split(Y = school1_mat_sig_var$G3, SplitRatio = 0.7)
trainSet_mat_2 <-
  subset(x = school1_mat_sig_var, sampleSplit_mat_2 == TRUE)
testSet_mat_2 <-
  subset(x = school1_mat_sig_var, sampleSplit_mat_2 == FALSE)
model_mat_2 <- lm(G3 ~ . , data = trainSet_mat_2)
summary(model_mat_2)
preds_mat_2 <- predict(model_mat_2, testSet_mat_2)
# RMSE = 4.2926 (Basically worse than using all features, significant features but not predictive)
rmse_mat_2 <-
  rmse(actual = testSet_mat_2$G3, predicted = preds_mat_2)

# Try a decision tree model for mathematics dataset
# Use all features as decision tree has internal feature selection through split results
library(rpart)
tree_model_mat_1 <- rpart(formula = G3 ~ .,
                          data = trainSet_mat_1,
                          method = "anova")
summary(tree_model_mat_1)
preds_mat_3 <- predict(tree_model_mat_1, testSet_mat_1)
# RMSE = 3.8887 (A little bit of improvement over linear models)
rmse_mat_3 <-
  rmse(actual = testSet_mat_1$G3, predicted = preds_mat_3)

# Retrieve optimal cp value based on cross-validated error
opt_index_mat_1 <- which.min(tree_model_mat_1$cptable[, "xerror"])
cp_opt_mat_1 <- tree_model_mat_1$cptable[opt_index_mat_1, "CP"]
# Prune the model (to optimized cp value)
tree_model_mat_2 <-
  prune(tree = tree_model_mat_1, cp = cp_opt_mat_1)
summary(tree_model_mat_2)
preds_mat_4 <- predict(tree_model_mat_2, testSet_mat_1)
# RMSE = 3.7850 (Slight improvement over unpruned decision tree earlier)
rmse_mat_4 <-
  rmse(actual = testSet_mat_1$G3, predicted = preds_mat_4)

# Try XGBoost model for mathematics dataset
# XGBoost uses matrix data so that we need to convert our data into the xgb matrix type
library(xgboost)
library(caret)
xgb_train_mat_x = data.matrix(trainSet_mat_1[, -31])
xgb_train_mat_y = trainSet_mat_1[, 31]
xgb_test_mat_x = data.matrix(testSet_mat_1[, -31])
xgb_test_mat_y = testSet_mat_1[, 31]

xgb_train_mat_1 = xgb.DMatrix(data = xgb_train_mat_x, label = xgb_train_mat_y)
xgb_test_mat_1 = xgb.DMatrix(data = xgb_test_mat_x, label = xgb_test_mat_y)
xgb_model_mat_1 = xgboost(data = xgb_train_mat_1,
                          max.depth = 2,
                          nrounds = 50)
# RMSE = 3.6388 (Best model thus far)
preds_mat_5 = predict(xgb_model_mat_1, xgb_test_mat_1)
rmse_mat_5 = rmse(actual = testSet_mat_1$G3, predicted = preds_mat_5)

###################################################################################################################

### Linear regression model for Portuguese dataset
# Train/test split is a random process, and seed ensures the randomization works the same on different computers
set.seed(42)
# Use default 70/30 train/test split
sampleSplit_por_1 <- sample.split(Y = school2$G3, SplitRatio = 0.7)
trainSet_por_1 <- subset(x = school2, sampleSplit_por_1 == TRUE)
testSet_por_1 <- subset(x = school2, sampleSplit_por_1 == FALSE)

# Train 1 model using all features and predict results
# Adjusted R-squared: 0.2975/Model p-value: 2.2e-16
# Only 6 features are significant using a 5% significance threshold (schoolMS, sexM, failures, schoolsupyes, higheryes, famrel)
model_por_1 <- lm(G3 ~ . , data = trainSet_por_1)
summary(model_por_1)
preds_por_1 <- predict(model_por_1, testSet_por_1)
# RMSE = 2.7717 (Better result than math dataset given G3 range is 0-20)
library(Metrics)
rmse_por_1 <-
  rmse(actual = testSet_por_1$G3, predicted = preds_por_1)

# Repeat above steps for linear regression model with the 6 significant variables
school2_por_sig_var <-
  school2[, c('school',
              'sex',
              'failures',
              'schoolsup',
              'higher',
              'famrel',
              'G3')]
sampleSplit_por_2 <-
  sample.split(Y = school2_por_sig_var$G3, SplitRatio = 0.7)
trainSet_por_2 <-
  subset(x = school2_por_sig_var, sampleSplit_por_2 == TRUE)
testSet_por_2 <-
  subset(x = school2_por_sig_var, sampleSplit_por_2 == FALSE)
model_por_2 <- lm(G3 ~ . , data = trainSet_por_2)
summary(model_por_2)
preds_por_2 <- predict(model_por_2, testSet_por_2)
# RMSE = 2.8148 (Basically worse than using all features, significant features but not predictive)
rmse_por_2 <-
  rmse(actual = testSet_por_2$G3, predicted = preds_por_2)

# Try a decision tree model for porheporics dataset
# Use all features as decision tree has internal feature selection through split results
tree_model_por_1 <- rpart(formula = G3 ~ .,
                          data = trainSet_por_1,
                          method = "anova")
summary(tree_model_por_1)
preds_por_3 <- predict(tree_model_por_1, testSet_por_1)
# RMSE = 3.1503 (Slight deprovement over linear models)
rmse_por_3 <-
  rmse(actual = testSet_por_1$G3, predicted = preds_por_3)

# Retrieve optimal cp value based on cross-validated error
opt_index_por_1 <- which.min(tree_model_por_1$cptable[, "xerror"])
cp_opt_por_1 <- tree_model_por_1$cptable[opt_index_por_1, "CP"]
# Prune the model (to optimized cp value)
tree_model_por_2 <-
  prune(tree = tree_model_por_1, cp = cp_opt_por_1)
summary(tree_model_por_2)
preds_por_4 <- predict(tree_model_por_2, testSet_por_1)
# RMSE = 2.8862 (Visible improvement over unpruned decision tree but still worse than linear models)
rmse_por_4 <-
  rmse(actual = testSet_por_1$G3, predicted = preds_por_4)

# Try XGBoost model for Portuguese dataset
# XGBoost uses matrix data so that we need to convert our data into the xgb matrix type
xgb_train_por_x = data.matrix(trainSet_por_1[, -31])
xgb_train_por_y = trainSet_por_1[, 31]
xgb_test_por_x = data.matrix(testSet_por_1[, -31])
xgb_test_por_y = testSet_por_1[, 31]

xgb_train_por_1 = xgb.DMatrix(data = xgb_train_por_x, label = xgb_train_por_y)
xgb_test_por_1 = xgb.DMatrix(data = xgb_test_por_x, label = xgb_test_por_y)
xgb_model_por_1 = xgboost(data = xgb_train_por_1,
                          max.depth = 2,
                          nrounds = 50)
# RMSE = 2.7432(Best model thus far)
preds_por_5 = predict(xgb_model_por_1, xgb_test_por_1)
rmse_por_5 = rmse(actual = testSet_por_1$G3, predicted = preds_por_5)