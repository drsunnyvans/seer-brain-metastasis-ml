library(readxl)
library(writexl)

# Dataset not provided due to data-use restrictions; place your dataset in data as brain_mets_dataset.xlsx
data - read_excel(databrain_mets_dataset.xlsx)
print("Data loaded successfully.")

# Convert variables to factors, except the target variable
predictors <- setdiff(names(data), "SEER Combined Mets at DX-brain (2010+)")
data[predictors] <- lapply(data[predictors], as.factor)
data$`SEER Combined Mets at DX-brain (2010+)` <- as.factor(data$`SEER Combined Mets at DX-brain (2010+)`)
print("Variables converted to factors.")

# Automatically identify categorical variables based on data type
categorical_columns <- sapply(data, function(x) is.factor(x) || is.character(x))

# Apply one-hot encoding to each categorical column
for (col in names(categorical_columns)[categorical_columns]) {
  # Check if the factor has less than 2 levels
  if (length(levels(data[[col]])) < 2) {
    print(paste("Skipping encoding for", col, "due to insufficient levels."))
    next  # Skip to the next iteration
  }

  print(paste("Encoding", col))
  dummies <- model.matrix(~ . - 1, data = data[col])
  data[[col]] <- NULL
  data <- cbind(data, dummies)
}

# Write the modified data back to an Excel file
write_xlsx(data, "Final Table2-Encoding.xlsx")
print("Encoded data written to Excel.")


