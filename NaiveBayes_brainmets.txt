library(data.table)
library(readxl)
library(naivebayes)
library(iml)
library(caret)
library(smotefamily)  # For SMOTE
library(openxlsx)  # For writing to Excel

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

# Separate features and target
features <- data[, setdiff(names(data), target_col), with = FALSE]
target <- data[[target_col]]

# Convert features to data.frame if not already
features <- as.data.frame(features)

# Initialize lists to store models and metrics
models <- list()
confusion_matrices <- list()

# Perform training 5 times
set.seed(123)  # For reproducibility
for (i in 1:5) {
  # Split data into training and testing sets
  train_index <- createDataPartition(target, p = 0.8, list = FALSE)
  features_train <- features[train_index, ]
  target_train <- target[train_index]
  features_test <- features[-train_index, ]
  target_test <- target[-train_index]

  # Apply SMOTE to balance the training dataset
  smote_data <- SMOTE(X = features_train, target = target_train, K = 5)
  features_balanced <- smote_data$data[, -ncol(smote_data$data)]
  target_balanced <- as.factor(smote_data$data[, ncol(smote_data$data)])

  # Train Naive Bayes model using the balanced training dataset
  model <- naive_bayes(x = features_balanced, y = target_balanced)
  models[[i]] <- model

  # Evaluate the model on the testing set
  predicted_probs <- predict(model, newdata = features_test, type = "prob")[, 2]

  # Assuming the positive class is the second level in the target_balanced factor
  positive_class <- levels(target_balanced)[2]
  negative_class <- levels(target_balanced)[1]

  # Adjust the threshold and predicted classes
  predicted_classes <- as.factor(ifelse(predicted_probs > 0.5, positive_class, negative_class))

  # Ensure factor levels match between predicted_classes and target_test
  predicted_classes <- factor(predicted_classes, levels = c(negative_class, positive_class))
  target_test <- factor(target_test, levels = c(negative_class, positive_class))

  # Generate the confusion matrix
  confusion_matrix <- confusionMatrix(predicted_classes, target_test, positive = positive_class)
  confusion_matrices[[i]] <- confusion_matrix
  print(confusion_matrix)
}

# Calculate average metrics
avg_metrics <- lapply(confusion_matrices, function(cm) cm$byClass)
avg_metrics <- Reduce("+", avg_metrics) / length(avg_metrics)
print(avg_metrics)

# Find the model with the highest sensitivity
sensitivities <- sapply(confusion_matrices, function(cm) cm$byClass['Sensitivity'])
best_model_index <- which.max(sensitivities)
best_model <- models[[best_model_index]]

# Define a predict function for the iml package
predict_function <- function(model, newdata) {
  probs <- predict(model, newdata = newdata, type = "prob")[, 2]
  matrix(probs, ncol = 1)
}

# Use SHAP for feature importance on the best model
# Create a predictor object for the iml package
predictor <- Predictor$new(
  model = best_model,
  data = features,
  y = target,
  predict.fun = predict_function
)

# Calculate SHAP values for the first observation
shapley <- Shapley$new(predictor, x.interest = features[1, , drop = FALSE])

# Extract SHAP values for the first observation
shap_values <- shapley$results

# Ensure shap_values is a data.table
shap_values <- as.data.table(shap_values)

# Create a mapping from one-hot encoded features back to original features
original_features <- colnames(data)[!colnames(data) %in% target_col]
one_hot_mapping <- setNames(original_features, make.names(original_features))

# Summarize SHAP values by original feature group
shap_values[, original_feature := sapply(feature, function(f) {
  # Find the corresponding original feature for each one-hot encoded feature
  matches <- grep(paste0("^", f), names(one_hot_mapping), value = TRUE)
  if (length(matches) > 0) {
    one_hot_mapping[matches[1]]
  } else {
    f
  }
})]

# Summarize SHAP values to get the importance of each original feature group
shap_summary <- shap_values[, .(shap_value_sum = sum(abs(phi))), by = original_feature]

# Save the SHAP summary to an Excel sheet
write.xlsx(shap_summary, file = "NBshap_summary.xlsx", rowNames = FALSE)

# Save the SHAP values to an Excel sheet
write.xlsx(shap_values, file = "NBshap_values.xlsx", rowNames = FALSE)

print("SHAP values have been calculated and saved.")
