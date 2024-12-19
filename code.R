# --------------------------------------------------------------
# Load Dataset and Packages
# --------------------------------------------------------------
install.packages("dplyr")
install.packages("car")
install.packages("e1071")
install.packages("caret")
install.packages("cluster")
install.packages("factoextra")
install.packages("fmsb")

df = read.csv('Dataset5_Automobile_data.csv')

library(fmsb)
library(dplyr)
library(car)
library(e1071)
library(caret)
library(cluster)  
library(factoextra)

attach(df)

# --------------------------------------------------------------
# Data Preprocessing
# --------------------------------------------------------------

## Missing Values Handling
missing_counts <- sapply(df, function(col) sum(col == "?" | is.na(col) | col == "", na.rm = TRUE))
missing_df <- data.frame(Column = names(missing_counts), Missing_Count = missing_counts)
missing_df

# Replace "?" with NA for easier handling
df[df == "?"] <- NA

# Remove rows with any missing values
df <- na.omit(df)

# --------------------------------------------------------------
# Drop Irrelevant Features
# --------------------------------------------------------------
df <- df[, !(names(df) %in% c("normalized.losses", "symboling"))]

# --------------------------------------------------------------
# Data Type Alignment
# --------------------------------------------------------------
str(df)

## Mapping for num.of.doors and num.of.cylinders
door_mapping <- c("two" = 2, "four" = 4)
df$num.of.doors <- door_mapping[df$num.of.doors]

cylinder_mapping <- c("two" = 2, "three" = 3, "four" = 4, "five" = 5, "six" = 6, "eight" = 8, "twelve" = 12)
df$num.of.cylinders <- cylinder_mapping[df$num.of.cylinders]

## Convert Columns from Character to Numeric
cols_to_convert <- c("bore", "stroke", "horsepower", "peak.rpm", "price")
df[cols_to_convert] <- lapply(df[cols_to_convert], function(x) as.numeric(as.character(x)))

# --------------------------------------------------------------
# Feature Engineering
# --------------------------------------------------------------

## Power-to-Weight Ratio
df <- df %>%
  mutate(power_to_weight_ratio = horsepower / `curb.weight`)

## Size Index
df <- df %>%
  mutate(size_index = `wheel.base` * width * height)

## Luxury Brand Indicator
df$make <- as.factor(df$make)
model <- lm(price ~ make, data = df)
summary(model)

luxury_brands <- c("jaguar", "mercedes-benz", "porsche", "bmw", "volvo")
df <- df %>%
  mutate(luxury_brand_indicator = ifelse(tolower(make) %in% luxury_brands, 1, 0)) %>%
  mutate(luxury_brand_indicator = as.factor(luxury_brand_indicator))

## Fuel Economy Difference
df <- df %>%
  mutate(fuel_economy_difference = highway.mpg - city.mpg)

## Performance Index
df <- df %>%
  mutate(performance_index = (engine.size * `compression.ratio` * peak.rpm) / 1000)

## Weight-to-Size Ratio
df <- df %>%
  mutate(weight_to_size_ratio = `curb.weight` / size_index)

## Compression Efficiency
df <- df %>%
  mutate(compression_efficiency = (`compression.ratio` * horsepower) / engine.size)

## Doors-to-Weight Ratio
df <- df %>%
  mutate(doors_to_weight_ratio = `num.of.doors` / `curb.weight`)

# --------------------------------------------------------------
# Multicollinearity & Feature Selection
# --------------------------------------------------------------

## Drop Redundant Features
df <- df %>%
  select(-wheel.base, -length, -width, -curb.weight, -engine.size, -highway.mpg, -city.mpg, -horsepower, -compression.ratio, -make)

## Calculate VIF Scores
numeric_vars <- sapply(df, is.numeric)
numeric_df <- df[, numeric_vars]
vif_scores <- vif(lm(price ~ ., data = numeric_df))
vif_df <- data.frame(Variable = names(vif_scores), VIF = vif_scores)
vif_df <- vif_df[order(-vif_df$VIF), ]
vif_df

# --------------------------------------------------------------
# Outlier Analysis
# --------------------------------------------------------------

## Detect Outliers Using IQR Method
detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  outliers <- x[x < lower_bound | x > upper_bound]
  return(length(outliers))
}

outlier_counts <- sapply(df[, sapply(df, is.numeric)], detect_outliers)
outlier_df <- data.frame(Column = names(outlier_counts), Outlier_Count = outlier_counts)
outlier_df

# --------------------------------------------------------------
# Boxplots for Selected Features
# --------------------------------------------------------------
par(mfrow = c(1, 2))
boxplot(df$fuel_economy_difference, main = "Fuel Economy Difference", ylab = "Fuel Economy Difference")
boxplot(df$performance_index, main = "Performance Index", ylab = "Performance Index")

# --------------------------------------------------------------
# Dummification of Character/Factor Columns
# --------------------------------------------------------------

## Dummify All Character/Factor Variables
char_factor_cols <- names(df)[sapply(df, function(x) is.character(x) | is.factor(x))]
for (col in char_factor_cols) {
  if (is.character(df[[col]])) {
    df[[col]] <- as.factor(df[[col]])
  }
  dummy_vars <- dummyVars(paste("~", col), data = df)
  dummy_df <- data.frame(predict(dummy_vars, newdata = df))
  df <- cbind(df, dummy_df)
  df <- df[, -which(names(df) == col)]
}

# --------------------------------------------------------------
# Standardization
# --------------------------------------------------------------
numeric_cols <- sapply(df, is.numeric)
df_numeric <- df[, numeric_cols]

# Standardize the numeric columns
df_scaled <- scale(df_numeric)

# Convert scaled data back to data frame
df_scaled <- as.data.frame(df_scaled)

str(df_scaled)

# --------------------------------------------------------------
# Feature Selection
# --------------------------------------------------------------

## Linear Regression Model
lm_model <- lm(price ~ ., data = df_scaled)

# Extract coefficients as feature importance
lm_importance <- summary(lm_model)$coefficients[, "Estimate"]
lm_importance <- data.frame(Feature = names(lm_importance), Importance = lm_importance)
print(lm_importance)

## Random Forest Model

# Load Random Forest Library
library(randomForest)

# Fit Random Forest Model
rf_model <- randomForest(price ~ ., data = df_scaled, importance = TRUE)

# Extract Feature Importance
rf_importance <- data.frame(Feature = rownames(rf_model$importance),
                            Importance = rf_model$importance[, "IncNodePurity"])
print(rf_importance)

# Plot Feature Importance
barplot(rf_importance$Importance, names.arg = rf_importance$Feature, las = 2, main = "Random Forest Feature Importance")

# --------------------------------------------------------------
# Final Feature Selection and Normalization
# --------------------------------------------------------------

# Normalize a Vector to Range 0-1
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Normalize Feature Importance from Linear Regression and Random Forest
lm_importance$Normalized_Importance <- normalize(abs(lm_importance$Importance))
rf_importance$Normalized_Importance <- normalize(rf_importance$Importance)

# Rename Columns for Clarity Before Merging
colnames(lm_importance) <- c("Feature", "Linear_Importance", "Linear_Normalized")
colnames(rf_importance) <- c("Feature", "RandomForest_Importance", "RandomForest_Normalized")

# Merge Importance from Both Models by Feature
combined_importance <- merge(lm_importance, rf_importance, by = "Feature", all = TRUE)

# Fill NA Values with 0 (In Case a Feature is Missing from One of the Models)
combined_importance[is.na(combined_importance)] <- 0

# Calculate Average Normalized Importance Across Methods
combined_importance$Average_Normalized_Importance <- rowMeans(
  combined_importance[, c("Linear_Normalized", "RandomForest_Normalized")])

# Sort by Average Normalized Importance
combined_importance <- combined_importance[order(-combined_importance$Average_Normalized_Importance), ]

# Select Top N Features
top_features <- head(combined_importance, 10)

# Print Combined Importance Table and Top Features
print(combined_importance)
print(top_features)

# Adjust Plot Size and Margins for Visualization
par(mar = c(12, 5, 4, 2))

# Plot Top Features
barplot(
  height = t(as.matrix(top_features[, c("Linear_Normalized", "RandomForest_Normalized")])),
  beside = TRUE,
  names.arg = top_features$Feature,
  las = 2,
  col = c("skyblue", "orange"),
  legend.text = c("Linear Regression", "Random Forest"),
  main = "Top Features Importance Comparison",
  ylab = "Normalized Importance",
  cex.names = 0.8
)

# --------------------------------------------------------------
# Luxury Score Construction
# --------------------------------------------------------------

## Luxury Score Calculation
calculate_luxury_score <- function(df, top_features) {
  # Extract Top Features and Their Normalized Importance
  selected_features <- top_features$Feature
  feature_weights <- top_features$Average_Normalized_Importance
  
  # Ensure Feature Weights Sum to 1
  feature_weights <- feature_weights / sum(feature_weights)
  
  # Create Luxury Score Column
  df$Luxury_Score <- rowSums(df[, selected_features] * feature_weights)
  
  return(df)
}

# Apply Function to Scaled Dataset
df_scaled_with_luxury_score <- calculate_luxury_score(df_scaled, top_features)

# Plot Distribution of Luxury Scores
hist(
  df_scaled_with_luxury_score$Luxury_Score,
  main = "Distribution of Luxury Scores",
  xlab = "Luxury Score",
  col = "lightblue",
  border = "black",
  breaks = 15
)

# Scatter Plot: Luxury Score vs Price
plot(
  df_scaled_with_luxury_score$Luxury_Score,
  df_scaled_with_luxury_score$price,
  main = "Luxury Score vs Price",
  xlab = "Luxury Score",
  ylab = "Price",
  col = "darkblue",
  pch = 19
)

# --------------------------------------------------------------
# Clustering Analysis
# --------------------------------------------------------------

## Subset Dataset for Clustering
clustering_features <- df_scaled_with_luxury_score[, c(
  "power_to_weight_ratio",
  "size_index",
  "luxury_brand_indicator.1",
  "fuel_economy_difference",
  "performance_index",
  "weight_to_size_ratio",
  "compression_efficiency",
  "doors_to_weight_ratio",
  "Luxury_Score",
  "price"
)]

# --------------------------------------------------------------
# Silhouette Analysis to Determine Optimal Clusters
# --------------------------------------------------------------

silhouette_analysis <- function(data, max_clusters = 10) {
  sil_width <- numeric(max_clusters - 1)
  
  for (k in 2:max_clusters) {
    kmeans_model <- kmeans(data, centers = k, nstart = 25)
    sil <- silhouette(kmeans_model$cluster, dist(data))
    sil_width[k - 1] <- mean(sil[, 3])
  }
  
  return(sil_width)
}

# Perform Silhouette Analysis
max_clusters <- 10
sil_width <- silhouette_analysis(clustering_features, max_clusters)

# Plot Silhouette Scores for Each k
plot(2:max_clusters, sil_width, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters (k)", ylab = "Average silhouette width",
     main = "Elbow Method: Optimal Clusters")
abline(v = which.max(sil_width) + 1, col = "red", lty = 2)

# Determine Optimal Number of Clusters
optimal_k <- which.max(sil_width) + 1
cat("Optimal number of clusters based on silhouette score:", optimal_k, "\n")

# --------------------------------------------------------------
# K-Means Clustering with Optimal Number of Clusters
# --------------------------------------------------------------

final_kmeans <- kmeans(clustering_features, centers = 3, nstart = 25)

# Visualize Clustering Results
library(factoextra)
fviz_cluster(final_kmeans, data = clustering_features, geom = "point",
             main = "Clustering Visualization (Optimal k)")

# --------------------------------------------------------------
# Cluster Analysis
# --------------------------------------------------------------

# Add Cluster Labels to Dataset
df_scaled_with_luxury_score$cluster <- final_kmeans$cluster

# Calculate Cluster-Level Statistics
cluster_stats <- df_scaled_with_luxury_score %>%
  group_by(cluster) %>%
  summarize(
    mean_luxury_score = mean(Luxury_Score),
    median_luxury_score = median(Luxury_Score),
    sd_luxury_score = sd(Luxury_Score),
    count = n()
  )

print(cluster_stats)

# --------------------------------------------------------------
# Visualization: Boxplot of Feature Distribution by Cluster
# --------------------------------------------------------------

# Select Feature for Boxplot (e.g., 'Luxury_Score')
feature_to_plot <- "Luxury_Score"

# Create Boxplot for Selected Feature by Cluster
ggplot(df_scaled_with_luxury_score, aes(x = factor(cluster), y = .data[[feature_to_plot]], fill = factor(cluster))) +
  geom_boxplot() +
  labs(title = paste("Distribution of", feature_to_plot, "by Cluster"),
       x = "Cluster",
       y = feature_to_plot,
       fill = "Cluster") +
  theme_minimal()

# --------------------------------------------------------------
# Radar Chart of Cluster Centers by Feature
# --------------------------------------------------------------

cluster_centers <- as.data.frame(final_kmeans$centers)
cluster_centers <- rbind(rep(max(cluster_centers), ncol(cluster_centers)),
                         rep(min(cluster_centers), ncol(cluster_centers)),
                         cluster_centers)

radarchart(cluster_centers,
           axistype = 1,
           pcol = c("red", "green", "blue"),
           pfcol = adjustcolor(c("#FF9999", "#99FF99", "#9999FF"), alpha.f = 0.3),
           plwd = 2,
           cglcol = "grey", cglty = 1, cglwd = 0.8,
           vlcex = 0.8,
           title = "Cluster Centers by Feature")

# --------------------------------------------------------------
# Cluster-Level Statistics for Specified Variables
# --------------------------------------------------------------

cluster_stats <- df_scaled_with_luxury_score %>%
  group_by(cluster) %>%
  summarize(
    across(
      c(
        "power_to_weight_ratio",
        "size_index",
        "luxury_brand_indicator.1",
        "fuel_economy_difference",
        "performance_index",
        "weight_to_size_ratio",
        "compression_efficiency",
        "doors_to_weight_ratio",
        "Luxury_Score",
        "price"
      ),
      list(mean = mean)
    ),
    count = n()
  )

print(cluster_stats)

# --------------------------------------------------------------
# Visualization: Feature Means by Cluster
# --------------------------------------------------------------

cluster_means <- tibble::tibble(
  cluster = c(1, 2, 3),
  power_to_weight_ratio_mean = c(-1.2112051, 1.1070278, -0.2719148),
  size_index_mean = c(1.0100569, 0.3585888, -0.3047082),
  luxury_brand_indicator_mean = c(-0.2277566, -0.9761798, 0.4399394),
  fuel_economy_difference_mean = c(-0.6336319, -0.1890356, 0.1762587),
  performance_index_mean = c(2.1013806, 0.6273455, -0.5847224),
  weight_to_size_ratio_mean = c(0.06967552, 1.16250959, -0.49277906),
  compression_efficiency_mean = c(2.53229070, -0.02752784, -0.37975287),
  doors_to_weight_ratio_mean = c(0.07208247, -0.70114069, 0.27958218),
  Luxury_Score_mean = c(0.06761755, 0.18050763, -0.09856919),
  price_mean = c(0.3484391, 1.0797236, -0.5015142)
)

cluster_means_melted <- melt(cluster_means, id.vars = "cluster")

ggplot(cluster_means_melted, aes(x = variable, y = value, fill = factor(cluster))) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Feature Means by Cluster",
       x = "Features",
       y = "Mean Value",
       fill = "Cluster") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
