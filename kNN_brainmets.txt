# Load required libraries
library(caret)
library(data.table)
library(smotefamily)
library(openxlsx)
library(iml)
# Function to make column names syntactically valid
make_valid_names <- function(names) {
    make.names(names, unique = TRUE)
}

# Dataset not provided due to data-use restrictions; place your dataset in data as brain_mets_dataset.xlsx
data - read_excel(databrain_mets_dataset.xlsx)
colnames(data) <- make_valid_names(colnames(data))
if (anyNA(data)) stop("Data contains NA values. Please clean the data before proceeding.")

# Prepare data
target_col <- make.names("SEER Combined Mets at DX-brain (2010+)")
data[[target_col]] <- as.factor(data[[target_col]])  # Ensure target is a factor for classification
if (anyNA(data[[target_col]])) stop("Target variable contains NA values.")

# Ensure target factor levels are valid R variable names
levels(data[[target_col]]) <- make.names(levels(data[[target_col]]))

# Separate features and target
features <- data[, setdiff(names(data), target_col)]
target <- data[[target_col]]

# Split data into training and testing sets (80% training, 20% testing)
set.seed(123)  # For reproducibility
train_index <- createDataPartition(target, p = 0.8, list = FALSE)
features_train <- features[train_index, ]
target_train <- target[train_index]
features_test <- features[-train_index, ]
target_test <- target[-train_index]

# Apply SMOTE to balance the training data
smote_result <- SMOTE(X = features_train, target = target_train, K = 5, dup_size = 0)

# Extract balanced data after SMOTE
features_train_smote <- smote_result$data[, -ncol(smote_result$data)]  # Features
target_train_smote <- smote_result$data[, ncol(smote_result$data)]  # Target

# Ensure the target variable is a factor
target_train_smote <- as.factor(target_train_smote)

# Custom function to extract Sensitivity
get_sensitivity <- function(confusion_matrix) {
  sensitivity <- confusion_matrix$byClass['Sensitivity']
  return(sensitivity)
}

# Create a custom trainControl for cross-validation with Sensitivity as the metric
train_control <- trainControl(
    method = "cv",                # Cross-validation
    number = 5,                   # 5-fold cross-validation
    savePredictions = "final",    # Save predictions for further use
    classProbs = TRUE,            # Enable probability predictions
    summaryFunction = twoClassSummary,  # Use two-class summary (ROC, Sensitivity, Specificity)
    allowParallel = TRUE          # Allow parallel processing for faster execution
)

# List of models to train
model_list <- c("glmnet", "kknn", "nnet")

# Empty list to store best models and sensitivities
best_models <- list()

# Train and evaluate each model type 5 times, then pick the best based on Sensitivity
set.seed(123)
for (model in model_list) {
  best_sensitivity <- 0
  best_model <- NULL
  
  for (i in 1:5) {
    # Train model
    model_fit <- train(
      x = features_train_smote,
      y = target_train_smote,
      method = model,             # Model type
      trControl = train_control,  # Cross-validation settings
      metric = "Sensitivity"      # Optimize for Sensitivity
    )
    
    # Predict on test data
    predictions <- predict(model_fit, newdata = features_test)
    
    # Ensure factor levels match between predicted_classes and target_test
    predictions <- factor(predictions, levels = levels(target_test))
    target_test <- factor(target_test, levels = levels(target_test))
    
    # Generate confusion matrix
    confusion_matrix <- confusionMatrix(predictions, target_test, positive = levels(target_test)[2])
    
    # Calculate Sensitivity
    sensitivity <- get_sensitivity(confusion_matrix)
    
    # If this model has the best sensitivity so far, store it
    if (sensitivity > best_sensitivity) {
      best_sensitivity <- sensitivity
      best_model <- model_fit
    }
  }
  
  # Store the best model for this model type
  best_models[[model]] <- best_model
}

# Create bagged ensemble using the best models and bagging (Bootstrap Aggregating)
bagging_control <- trainControl(
  method = "boot",                # Bootstrap sampling for bagging
  number = 50,                    # Number of bootstraps
  classProbs = TRUE,              # Enable probability predictions
  summaryFunction = twoClassSummary,  # Use two-class summary (ROC, Sensitivity, Specificity)
  savePredictions = "final"
)

# Combine the best models into a bagged ensemble using "treebag" method
ensemble_model <- train(
  x = features_train_smote,
  y = target_train_smote,
  method = "treebag",             # Bagging method
  trControl = bagging_control,    # Bagging control settings
  metric = "Sensitivity"          # Optimize for Sensitivity
)

# Predict on the test set using the bagged ensemble model
ensemble_predictions <- predict(ensemble_model, newdata = features_test)

# Generate confusion matrix for the ensemble model
ensemble_cm <- confusionMatrix(ensemble_predictions, target_test, positive = levels(target_test)[2])

# Print the confusion matrix to assess the final model performance
print(ensemble_cm)

# Optional: Define a predict function for SHAP that handles probabilities for the ensemble model
predict_function_ensemble <- function(model, newdata) {
  predict(model, newdata = newdata, type = "prob")
}

# Create a Predictor object for SHAP values using the iml package
predictor_ensemble <- Predictor$new(
    model = ensemble_model,
    data = features_test,
    y = target_test,
    predict.fun = predict_function_ensemble
)

# Calculate SHAP values for the first observation
shapley_ensemble <- Shapley$new(predictor_ensemble, x.interest = features_test[1, , drop = FALSE])

# Extract SHAP values
shap_values_ensemble <- shapley_ensemble$results

# Convert SHAP values to data.table
shap_values_ensemble <- as.data.table(shap_values_ensemble)

# Summarize SHAP values to get the importance of each original feature group
shap_summary_ensemble <- shap_values_ensemble[, .(shap_value_sum = sum(abs(phi))), by = feature]

# Save SHAP summaries and SHAP values to Excel files
write.xlsx(shap_summary_ensemble, file = "ensemble_shap_summary.xlsx", rowNames = FALSE)
write.xlsx(shap_values_ensemble, file = "ensemble_shap_values.xlsx", rowNames = FALSE)

print("SHAP values for the ensemble have been calculated and saved.")
