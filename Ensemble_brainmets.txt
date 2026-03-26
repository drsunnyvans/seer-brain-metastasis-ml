# Load required libraries
library(data.table)
library(caret)
library(caretEnsemble)
library(iml)
library(openxlsx)
library(smotefamily)
library(readxl)  # For SMOTE

# Function to make column names syntactically valid
make_valid_names <- function(names) {
    make.names(names, unique = TRUE)
}

# Dataset not provided due to data-use restrictions; place your dataset in /data as "brain_mets_dataset.xlsx"
data <- read_excel("data/brain_mets_dataset.xlsx")
colnames(data) <- make_valid_names(colnames(data))
if (anyNA(data)) stop("Data contains NA values. Please clean the data before proceeding.")

# Prepare data
target_col <- make.names("SEER Combined Mets at DX-brain (2010+)")
data[[target_col]] <- as.factor(data[[target_col]])  # Ensure target is a factor for classification
if (anyNA(data[[target_col]])) stop("Target variable contains NA values.")

# Ensure target factor levels are valid R variable names
levels(data[[target_col]]) <- make.names(levels(data[[target_col]]))

# Separate features and target
features <- data[, setdiff(names(data), target_col), with = FALSE]
target <- data[[target_col]]

# Convert features to data.frame if not already
features <- as.data.frame(features)

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

# Create a custom trainControl for cross-validation
train_control <- trainControl(
    method = "cv",                 # Cross-validation
    number = 5,                    # 5-fold cross-validation
    savePredictions = "final",     # Save predictions for stacking
    classProbs = TRUE,             # Enable probability predictions
    summaryFunction = twoClassSummary  # To calculate metrics like ROC, Sensitivity, etc.
)

# Define model training using caretList
model_list <- caretList(
    x = features_train_smote, 
    y = target_train_smote,
    trControl = train_control,      # Cross-validation settings
    methodList = c("glmnet", "kknn", "nnet"),
    tuneList = NULL                 # You can add specific tuning grids for each model here if needed
)

# Create an ensemble model using caretEnsemble
ensemble_model <- caretEnsemble(
    model_list, 
    metric = "Sensitivity",            # Metric to optimize (you can change this if needed)
    trControl = train_control
)

# Predict probabilities on the test set using the ensemble model
probabilities <- predict(ensemble_model, newdata = features_test)

# Extract probabilities for the positive class (second column)
positive_class_probabilities <- probabilities[, 2]

# Convert probabilities to class labels using a threshold of 0.5
predictions <- ifelse(positive_class_probabilities > 0.5, levels(target_test)[2], levels(target_test)[1])

# Ensure predictions are factors with the same levels as the target
predictions <- factor(predictions, levels = levels(target_test))

# Generate a confusion matrix with all metrics
confusion_matrix <- confusionMatrix(predictions, target_test, mode = "everything", positive = levels(target_test)[2])
print(confusion_matrix)

# Define a predict function for SHAP that handles probabilities for the ensemble model
predict_function_ensemble <- function(model, newdata) {
  # Directly use predict without setting the type argument
  predict(model, newdata = newdata)
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
