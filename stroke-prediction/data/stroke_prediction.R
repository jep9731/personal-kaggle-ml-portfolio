# Set-up environment ------------------------------------------------------
# clear environment
rm(list = ls())

# Install packages if not already installed
## Metrics
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret") #AUC
if (!requireNamespace("Metrics", quietly = TRUE)) install.packages("Metrics") # Recall
if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC") # ROC

## Visualizations
if (!requireNamespace("corrplot", quietly = TRUE)) install.packages("corrplot")

## Models
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("xgboost", quietly = TRUE)) install.packages("xgboost")

# Load libraries
library(tidyverse)
library(randomForest)
library(xgboost)
library(caret)
library(Metrics)
library(pROC)
library(corrplot)

# Load dataset
stroke_df <- read.csv("C:\\Users\\joshp\\OneDrive\\Documents\\Kaggle\\Stroke Prediction\\data\\healthcare-dataset-stroke-data.csv")

# EDA ---------------------------------------------------------------------
# Frequency tables
## Gender
stroke_df %>%
  group_by(gender) %>%
  summarise(
    count = n(), .groups = 'drop'
  )

## Work type
stroke_df %>%
  group_by(work_type) %>%
  summarise(
    count = n(), .groups = 'drop'
  )

## Residence type
stroke_df %>%
  group_by(Residence_type) %>%
  summarise(
    count = n(), .groups = "drop"
  )

## Married
stroke_df %>%
  group_by(ever_married) %>%
  summarise(
    count = n(), .groups = "drop"
  )

## Smoked
stroke_df %>%
  group_by(smoking_status) %>%
  summarise(
    count = n(), .groups = "drop"
  )

# Bivariate analysis
## Gender & heart disease
table(stroke_df$gender, stroke_df$heart_disease)

## Gender & hypertension
table(stroke_df$gender, stroke_df$hypertension)

## Work type & heart disease
table(stroke_df$work_type, stroke_df$heart_disease)

## Work type & hypertension
table(stroke_df$work_type, stroke_df$hypertension)

## Married & heart disease
table(stroke_df$ever_married, stroke_df$heart_disease)

## Married & hypertension
table(stroke_df$ever_married, stroke_df$hypertension)

## Smoking status & heart disease
table(stroke_df$smoking_status, stroke_df$heart_disease)

## Smoking status & hypertension
table(stroke_df$smoking_status, stroke_df$hypertension)

# Correlation values
correlation_matrix <- round(
  cor(stroke_df[, c("stroke", "age", "hypertension", 
                    "heart_disease", "avg_glucose_level"
                    )]
  ), 
  2
)

print(correlation_matrix)

# Plot correlation values
corrplot(correlation_matrix, 
         type = "upper", 
         order = "hclust",
         tl.col = "black", 
         tl.srt = 45)

### Stroke had a very small relationship with all the numeric variables, with age being 
### at .25. All other variables have a .13 correlation value with stroke.

# Distribution of age
ggplot(stroke_df, aes(x = age)) +
  geom_density(fill = "blue", alpha = 0.7) +
  labs(title = "Density of Age", 
       x = "Age", 
       y = "Density") +
  theme_bw()

# Distribution of glucose level
ggplot(stroke_df, aes(x = avg_glucose_level)) +
  geom_histogram(binwidth = 10, fill = "blue", alpha = 0.7, color = "white") +
  labs(title = "Histogram for Average Glucose Level", 
       x = "Average Glucose Level", 
       y = "Frequency") +
  theme_bw()

# Create a bar plot of the binary variables
## Hypertension
ggplot(stroke_df, aes(x = factor(hypertension))) +
  geom_bar(stat = "count", color = "blue", fill = "blue", width = 0.5) +
  geom_text(stat = "count", size = 5, aes(label = after_stat(count)),
            color = "white", position=position_stack(vjust=0.5)
  ) +
  labs(title = "Hypertension Class Distribution (Binary)", 
       x = "Class", 
       y = "Count") +
  scale_x_discrete(
    breaks = c(0, 1),
    labels = c("No", "Yes")
  ) +
  theme_bw()

## Heart Disease
ggplot(stroke_df, aes(x = factor(heart_disease))) +
  geom_bar(stat = "count", color = "blue", fill = "blue", width = 0.5) +
  geom_text(stat = "count", size = 5, aes(label = after_stat(count)),
            color = "white", position=position_stack(vjust=0.5)
  ) +
  labs(title = "Heart Disease Class Distribution (Binary)", 
       x = "Class", 
       y = "Count") +
  scale_x_discrete(
    breaks = c(0, 1),
    labels = c("No", "Yes")
  ) +
  theme_bw()

# Data preparation ---------------------------------------------------------------------
# Get information on data
str(stroke_df)

# Change BMI to numeric
stroke_df$bmi <- as.numeric(stroke_df$bmi)

# Get missing data values
sapply(stroke_df, function(x) sum(is.na(x)))

# Only BMI column has missing values so that is the only column that needs imputation, 
# which we will use the median as it is less likely to be affected by extreme values.

# Impute data
stroke_imputed <- stroke_df %>%
  mutate(
    bmi = replace_na(bmi, median(bmi, na.rm = TRUE))
  )

# One hot encoding
## Identify character columns
char_col_names <- stroke_imputed %>% 
  select(where(is.character)) %>% 
  names()

## Create dummy variables and drop one level for each factor
dummy_vars <- dummyVars(~ ., 
                        data = stroke_imputed[char_col_names], 
                        fullRank = TRUE
                        )

## Generate one-hot encoded data
ohe_df <- as.data.frame(predict(dummy_vars, 
                             newdata = stroke_imputed[char_col_names])
                     )

## Remove original character columns and bind dummies
stroke_encoded <- stroke_imputed %>%
  select(-all_of(char_col_names)) %>%
  bind_cols(ohe_df)

## Make sure names are accurate
names(stroke_encoded) <- make.names(names(stroke_encoded))

## Check structure
str(stroke_encoded) # Col names look good

# Scale data
## Identify numeric columns
num_col_names <- c("age", "avg_glucose_level", "bmi")

## Create copy and scale
stroke_scaled <- stroke_encoded
stroke_scaled[num_col_names] <- scale(stroke_encoded[num_col_names])

## Make sure names are accurate
names(stroke_scaled) <- make.names(names(stroke_scaled))

## Check structure
str(stroke_scaled) # Col names look good

### Our dataset has been cleaned and now is ready for splitting
### and modeling. We will check the class for our target variable
### 'stroke' as we will likely need to address a class imbalance

# Removing outliers -------------------------------------------------------
# Calculate Q1, Q3 and IQR for numeric columns
remove_outliers <- function(df, col_name) {
  Q1 <- quantile(df[[col_name]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[col_name]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  
  # Define lower and upper bounds
  lower <- Q1 - 1.5 * IQR
  upper <- Q3 + 1.5 * IQR
  
  # Filter rows within bounds
  df <- df[df[[col_name]] >= lower & df[[col_name]] <= upper, ]
  return(df)
}

# Apply to stroke_scaled
stroke_clean <- stroke_scaled %>%
  remove_outliers("avg_glucose_level") %>%
  remove_outliers("bmi")

# Check dimensions before and after
cat("Original rows:", nrow(stroke_scaled), "\n")
cat("Rows after outlier removal:", nrow(stroke_clean), "\n")

# Splitting Data ----------------------------------------------------------
# Set seed
set.seed(123)

# Calculate sample size for training data (80%)
sample_size <- floor(0.8 * nrow(stroke_clean))

# Subset dataframe
train_ind <- sample(seq_len(nrow(stroke_clean)), size = sample_size)

# Create train and test datasets
train_data <- stroke_clean[train_ind, ]
test_data <- stroke_clean[-train_ind, ]

# Remove "id" columns
train_data <- subset(train_data, select = -id)
test_data  <- subset(test_data,  select = -id)

# Ensure target variable is a factor
train_data$stroke <- factor(train_data$stroke, levels = c(0, 1))
test_data$stroke  <- factor(test_data$stroke, levels = c(0, 1))

# Check shape
print(dim(train_data))
print(dim(test_data))

# Class imbalance ---------------------------------------------------------
# Determine class imbalance (get the counts of each class)
class_counts <- table(train_data$stroke)
print("Class Counts:")
print(class_counts)

# Get the proportions (percentages) of each class
class_proportions <- prop.table(class_counts) * 100
print("Class Proportions (%):")
print(class_proportions)

# Visualize imbalance
ggplot(train_data, aes(x = stroke)) +
  geom_bar(stat = "count", color = "blue", fill = "blue", width = 0.5) +
  geom_text(stat = "count", 
            size = 5, aes(label = after_stat(count)),
            color = "white", position=position_stack(vjust=0.5)
  ) +
  labs(title = "Stroke Class Distribution (Target)", 
       x = "Class", 
       y = "Count") +
  scale_x_discrete(
    breaks = c(0, 1),
    labels = c("No", "Yes")
  ) +
  theme_bw()

### There seems to be a large class imbalance between
### people that have NOT had a stroke versus people
### that have had a stroke. We will use the inverse
### frequency to balance the classes to prevent 
### the models from being biases.

# Calculate class weights
total_samples <- sum(class_counts) # Find total samples
num_classes <- length(class_counts) # Find number of classes
class_weights <- list(
  '0' = total_samples / (num_classes * class_counts['0']), # Get weight of '0' class
  '1' = total_samples / (num_classes * class_counts['1']) # Get weight of '1' class
)
print(class_weights) # print class weights

### We have our class weights so we can now set up the models.
### I will first run a logistic regression model, then a
### random forest model, and finally a XGBoost model.
### For each model, I will output the accuracy of the testing data,
### plot an ROC curve, and print the variables. Lastly, I will 
### compare the ROC and AUC prediction values of each model to 
### determine the best performing model.

# User-defined models -----------------------------------------------------
## Print metrics
print_metrics <- function(model, train, test, response = "stroke") {
  
  # Get predicted probabilities
  if (inherits(model, "glm")) {
    # Logistic regression
    probs <- predict(model, newdata = test, type = "response")
    
  } else if (inherits(model, "randomForest")) {
    # Random Forest
    probs <- predict(model, newdata = test, type = "prob")[, 2]
    
  } else if (inherits(model, "xgb.Booster")) {
    # XGBoost (matrix input required)
    dtest <- as.matrix(test[, setdiff(names(test), response)])
    probs <- predict(model, dtest)
    
  } else {
    stop("Unsupported model type")
  }
  
  # Class predictions
  preds <- ifelse(probs >= 0.3, 1, 0)
  
  # Metrics 
  roc_obj <- roc(test[[response]], probs)
  auc_val <- as.numeric(auc(roc_obj))
  cm <- table(
    Predicted = factor(preds, levels = c(0, 1)),
    Actual    = factor(test[[response]], levels = c(0, 1))
  )
  
  accuracy <- mean(preds == test[[response]])
  precision <- sum(preds == 1 & test[[response]] == 1) / sum(preds == 1)
  recall <- sum(preds == 1 & test[[response]] == 1) / sum(test[[response]] == 1)
  
  # Output
  cat("Confusion Matrix:\n")
  print(cm)
  
  cat("Metrics:\n")
  cat("Accuracy:", round(accuracy, 3), "\n")
  cat("Precision:", round(precision, 3), "\n")
  cat("Recall:", round(recall, 3), "\n")
  cat("AUC:", round(auc_val, 3), "\n")
  
  # ---- Return values ----
  return(list(
    auc = auc_val,
    recall = recall,
    accuracy = accuracy,
    precision = precision
  ))
}

## Plot ROC curve
plot_roc_curve <- function(model, test, response = "stroke", model_name = NULL,
                           save_path = NULL, width = 6, height = 6,
                           dpi = 300) {
  # Predicted probabilities 
  if (inherits(model, "glm")) {
    probs <- predict(model, newdata = test, type = "response")
    
  } else if (inherits(model, "randomForest")) {
    probs <- predict(model, newdata = test, type = "prob")[, 2]
    
  } else if (inherits(model, "xgb.Booster")) {
    dtest <- as.matrix(test[, setdiff(names(test), response)])
    probs <- predict(model, dtest)
    
  } else {
    stop("Unsupported model type")
  }
  
  # ROC & AUC
  roc_obj <- roc(test[[response]], probs, quiet = TRUE)
  auc_val <- auc(roc_obj)
  
  # Open device if saving
  if (!is.null(save_path)) {
    if (grepl("\\.pdf$", save_path)) {
      pdf(save_path, width = width, height = height)
    } else {
      png(save_path, width = width, height = height, units = "in", res = dpi)
    }
  }
  
  # Plot 
  plot(
    NULL,
    col = "#2C7BB6",
    lwd = 2,
    xlim = c(0, 100), ylim = c(0, 100),
    xlab = "False Positive Percentage",
    ylab = "True Positive Percentage",
    main = ifelse(
      is.null(model_name),
      paste("ROC Curve (AUC =", round(auc_val, 3), ")"),
      paste(model_name, "- ROC Curve (AUC =", round(auc_val, 3), ")")
    )
  )
  
  abline(a = 0, b = 1, lty = 2, col = "gray")
  
  # Scale to percentages for plotting
  roc_fp <- roc_obj$specificities * 100
  roc_tp <- roc_obj$sensitivities * 100
  lines(100 - roc_fp, roc_tp, col = "#D7191C", lwd = 2)
  
  # Close device
  if (!is.null(save_path)) dev.off()
}

## Plot multiple ROC curves
plot_multiple_roc <- function(models, labels, test, response = "stroke",
                              save_path = NULL,width = 6, height = 6,
                              dpi = 300) {
  n <- length(models)
  # Open device if saving
  if (!is.null(save_path)) {
    if (grepl("\\.pdf$", save_path)) {
      pdf(save_path, width = width, height = height)
    } else {
      png(save_path, width = width, height = height, units = "in", res = dpi)
    }
  }
  
  # Plot
  colors <- c("#2C7BB6", "#D7191C", "#1A9641")
  
  plot(NULL, xlim = c(0, 100), ylim = c(0, 100),
       xlab = "False Positive Rate",
       ylab = "True Positive Rate",
       main = "ROC Curve Comparison")
  
  abline(a = 0, b = 1, lty = 2, col = "gray")
  
  auc_vals <- numeric(n)
  
  for (i in seq_along(models)) {
    model <- models[[i]]
    
    if (inherits(model, "glm")) {
      probs <- predict(model, test, type = "response")
    } else if (inherits(model, "randomForest")) {
      probs <- predict(model, test, type = "prob")[, 2]
    } else if (inherits(model, "xgb.Booster")) {
      # Select same features as x_train
      test_mat <- as.matrix(test[, colnames(x_train)])
      probs <- predict(model, test_mat)
    } else {
      stop("Model type not supported")
    }
    
    roc_obj <- roc(test[[response]], probs, quiet = TRUE)
    auc_vals[i] <- auc(roc_obj)
    
    # Scale to percentages
    roc_fp <- roc_obj$specificities * 100
    roc_tp <- roc_obj$sensitivities * 100
    lines(100 - roc_fp, roc_tp, col = colors[i], lwd = 2)
  }
    # Add AUC text
    legend("bottomright",
           legend = sprintf("%s (AUC = %.3f)", labels, auc_vals),
           col = colors[1:n],
           lwd = 2,
           bty = "n")
  
  # Close device 
  if (!is.null(save_path)) dev.off()
}

# Logistic Regression Model ------------------------------------------------------------------
# Create observed weights
obs_weights <- if_else(
  train_data$stroke == 1,
  class_weights[['1']],
  class_weights[['0']]
)

# Fit model
log_model <- glm(
  stroke ~ .,
  data = train_data,
  family = binomial(),
  weights = obs_weights
)

# Print accuracy
log_results <- print_metrics(
  model = log_model, 
  train = train_data, 
  test = test_data
  )

# Save metrics
log_auc <- log_results$auc
log_recall <- log_results$recall

# Plot ROC curve
plot_roc_curve(
  model = log_model, 
  test = test_data,
  model_name = "Logistic Regression"
  )

# Save plot
plot_roc_curve(
  model = log_model,
  test = test_data,
  model_name = "Logistic Regression", 
  save_path = "C:\\Users\\joshp\\OneDrive\\Documents\\Kaggle\\Stroke Prediction\\figures\\logistic_roc.png" # change path
  )
               
# Show variables
log_vars <- varImp(log_model, scale = TRUE)
print(log_vars)

# Save variable names
log_vars_df <- as.data.frame(log_vars)
log_vars_df <- log_vars_df[order(-log_vars_df$Overall), , drop = FALSE] # Sort by descending order

# Get top 5 variable names
top_vars <- rownames(head(log_vars_df, n =5))

# Save to a vector
log_var_names <- top_vars
log_var_names

# RF Model ----------------------------------------------------------------
# Change class weights
classwt <- c(
  as.numeric(class_weights[['0']]),
  as.numeric(class_weights[['1']])
)

names(classwt) <- levels(train_data$stroke)

# Fit model
rf_model <- randomForest(
  stroke ~ .,
  data = train_data,
  ntree = 500,
  classwt = classwt,
  importance = TRUE
)

# Print accuracy
rf_results <- print_metrics(
  model = rf_model, 
  train = train_data, 
  test = test_data
  )
rf_results

# Save metrics
rf_auc <- rf_results$auc
rf_recall <- rf_results$recall

# Plot ROC curve
plot_roc_curve(
  model = rf_model, 
  test = test_data, 
  model_name = "Random Forest"
  )

# Save plot
plot_roc_curve(
  model = rf_model,
  test = test_data,
  model_name = "Random Forest",
  save_path = "C:\\Users\\joshp\\OneDrive\\Documents\\Kaggle\\Stroke Prediction\\figures\\rf_roc.png" # Change path
)

# Show variables
rf_vars <- importance(rf_model)
print(rf_vars)

# Convert to data frame and sort
rf_var_df <- as.data.frame(rf_vars)
rf_var_df <- rf_var_df[order(-rf_var_df$MeanDecreaseGini), , drop = FALSE]

# Get top variable names
rf_var_names <- rownames(head(rf_var_df, n = 5))
rf_var_names

# XGB Model ---------------------------------------------------------------
# Make labels for XGBoost model
train_label <- as.numeric(as.character(train_data$stroke))
test_label  <- as.numeric(as.character(test_data$stroke))

# Make matrices for xgboost
x_train <- model.matrix(stroke ~ ., train_data)[, -1]
x_test  <- model.matrix(stroke ~ ., test_data)[, -1]

dtrain <- xgb.DMatrix(
  data = x_train,
  label = train_label,
  weight = obs_weights
)

dtest <- xgb.DMatrix(
  data = x_test,
  label = test_label
)

# Create parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = sum(train_label == 0) / sum(train_label == 1)
)

# Fit model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  evals = list(
    train = dtrain,
    test  = dtest
  )
)

# Print accuracy
xgb_results <- print_metrics(
  model = xgb_model, 
  train = train_data, 
  test = test_data
  )
xgb_results

# Save metrics
xgb_auc <- xgb_results$auc
xgb_recall <- xgb_results$recall

# Plot ROC curve
plot_roc_curve(
  model = xgb_model,
  test = test_data,
  model_name = "XGBoost"
)

# Save plot
plot_roc_curve(
  model = xgb_model,
  test = test_data,
  model_name = "XGBoost",
  save_path = "C:\\Users\\joshp\\OneDrive\\Documents\\Kaggle\\Stroke Prediction\\figures\\xgb_roc.png"
)

# Show variables
importance_matrix <- xgb.importance(
  feature_names = colnames(x_train),
  model = xgb_model
  )

# View it
print(importance_matrix)

# Plot importance
xgb.plot.importance(importance_matrix, top_n = 10)

# Save variable names
xgb_var_names <- head(importance_matrix$Feature, n =5)

# Model Comparison --------------------------------------------------------
# Table of metrics
results_df <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  ROC_AUC = c(log_auc, rf_auc, xgb_auc),
  Recall = c(log_recall, rf_recall, xgb_recall)
)
results_df

## Save table
write.csv(
  results_df, "C:\\Users\\joshp\\OneDrive\\Documents\\Kaggle\\Stroke Prediction\\results\\results.csv"
)

# Combined ROC plot
## Lists for combined plot
model_list <- list(log_model, rf_model, xgb_model)
label_list <- c("Logistic Regression", "Random Forest", "XGBoost")

## Plot curve
plot_multiple_roc(
  models = model_list,
  labels = label_list,
  test = test_data
  )

## Save curve
plot_multiple_roc(
  models = model_list,
  labels = label_list,
  test = test_data,
  save_path = "C:\\Users\\joshp\\OneDrive\\Documents\\Kaggle\\Stroke Prediction\\figures\\model_comparions_roc.png"
)

# Table of variables
vars_df <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Variables = c(
    paste(log_var_names, collapse = ", "),
    paste(rf_var_names, collapse = ", "),
    paste(xgb_var_names, collapse = ", ")
  ),
  stringsAsFactors = FALSE
)

print(vars_df)
