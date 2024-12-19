# How to Quantify Automotive Luxury? 🚗✨

---

## Technologies and Skills
### Technologies Used:
- 📊 **Programming Language**: R
- 📚 **Libraries and Tools**: dplyr, ggplot2, caret, randomForest, factoextra, fmsb
- 📈 **Statistical Models**: Linear Regression, Random Forest
- 🧩 **Clustering Techniques**: K-Means Clustering

### Skills Demonstrated:
- 🧹 **Data Preprocessing**: Cleaning, transforming, and encoding complex datasets.
- 🧪 **Feature Engineering**: Creating meaningful metrics like Luxury Score and performance indices.
- 🔍 **Clustering Analysis**: Segmenting cars into actionable market groups.
- 📊 **Visualization**: Creating informative plots to support findings.
- 💡 **Business Insight Generation**: Translating data-driven results into actionable recommendations.
- 🎤 **Communication**: Presenting insights clearly through structured reports and code.

---

## Overview

### How Can Audi Redefine Luxury? 🏎️

Imagine you’re Audi’s product manager, about to design the next flagship model. What makes a car truly luxurious? Is it raw horsepower, sleek design, or the feel of the steering wheel in your hands? How do you measure something as intangible as "luxury"?  

This project transforms the concept of automotive luxury into a **Luxury Score**—a data-driven metric that quantifies luxury by combining critical features like **performance**, **design**, and **brand identity**. 

Let’s say Audi is planning to launch a new electric SUV. Using the Luxury Score, Audi can pinpoint which features—like a **high power-to-weight ratio**, **spacious interior dimensions**, or even the **fuel economy difference between city and highway driving**—matter most to their target audience. By balancing these attributes, Audi ensures its new model not only meets but exceeds customer expectations.  

What’s more, clustering analysis reveals **three distinct buyer segments**:
- **Performance enthusiasts** craving high speed and cutting-edge engineering. ⚡
- **Practical buyers** prioritizing comfort and affordability. 💼
- **Eco-conscious urbanites** who value efficiency and smart design. 🌱

With this insight, Audi can strategically design and market its car to dominate the competition. Whether it’s a high-performance RS model or an eco-friendly city cruiser, this data-driven approach helps Audi align its luxury DNA with what truly matters to its audience.  

So, next time you see an Audi cruising by, ask yourself: is this luxury redefined? The answer, now, lies in the data. 🚗✨

---

## Business Insights and Impacts
### Key Implications:
1. **Luxury Score Utility**: 
   - Provides automakers with a quantitative measure of luxury, enabling benchmarking against competitors. 📏
   - Helps identify areas for improvement in design, performance, or branding.

2. **Cluster-Based Marketing**:
   - **Cluster 1 (Mid-Range Practical Cars)**: Focus marketing efforts on reliability, balanced performance, and affordability. 💼
   - **Cluster 2 (High-Performance Luxury Cars)**: Highlight advanced engine technology, size, and luxury branding to appeal to premium buyers. 🏎️
   - **Cluster 3 (Economical Compact Cars)**: Emphasize fuel efficiency, practicality, and low cost to attract budget-conscious consumers. 🌱

3. **Strategic Product Development**:
   - Use insights from feature importance to refine existing models or design new ones tailored to customer preferences. 🔧
   - Enhance the perception of luxury by improving features like size, performance, and brand reputation.

4. **Market Positioning**:
   - Automotive brands can adjust their positioning strategies based on the clusters and their luxury scores to align with target demographics.

5. **Customer Insights**:
   - Empower customers with data-driven recommendations to select cars based on their priorities, such as performance, luxury, or cost-effectiveness. 🎯

---

## Objectives
- **Primary Goal**: Quantify automotive luxury by calculating a composite score based on key vehicle attributes. 🏆
- **Secondary Goal**: Identify natural groupings among car models to guide market segmentation and product strategy. 📊

---

## Dataset
The dataset contains information on 193 car models, with 39 features including:
- **Numerical Features**: Horsepower, engine size, curb weight, etc.
- **Categorical Features**: Make, fuel type, number of doors, etc.

### Key Engineered Features:
1. **Power-to-Weight Ratio**: Reflects performance relative to car weight. ⚖️
2. **Size Index**: Captures the overall dimensions of the car.
3. **Luxury Brand Indicator**: Binary indicator for brands like Jaguar, BMW, and Mercedes-Benz. ⭐
4. **Fuel Economy Difference**: Highway MPG - City MPG. 🚗
5. **Performance Index**: Combines engine size, compression ratio, and peak RPM. ⚡
6. **Weight-to-Size Ratio**: Relates car weight to its dimensions.
7. **Compression Efficiency**: Balances compression ratio, horsepower, and engine size.

---

## Methodology

### 1. Data Preprocessing
- Removed missing values and redundant features. 🧹
- Standardized numerical features and normalized engineered metrics.
- Encoded categorical variables (e.g., fuel type) into dummy variables.

### 2. Luxury Score Construction
- Combined top features identified through **Linear Regression** and **Random Forest** models. 🌟
- Weighted features based on their average importance across both models.
- Normalized feature values to compute a **Luxury Score** for each car.

### 3. Clustering Analysis
- Applied **K-Means Clustering** to segment cars into three distinct clusters. 🧩
- Used silhouette analysis to determine the optimal number of clusters.
- Clusters included:
  - **Cluster 1**: Mid-Range Practical Cars
  - **Cluster 2**: High-Performance Luxury Cars
  - **Cluster 3**: Economical Compact Cars

---

## Repository Structure
- **`Automobile.R`**: R script containing the complete analysis, from data preprocessing to clustering. 🖥️
- **`Dataset5_Automobile_data.csv`**: The cleaned and preprocessed dataset used for analysis. 📂
- **`Final_Wan.pdf`**: A detailed report summarizing the project methodology, results, and business insights. 📄

---

## Future Improvements
- Integrate additional datasets for better model robustness (e.g., sales data). 📊
- Explore advanced clustering techniques like hierarchical clustering. 🧠
- Develop an interactive dashboard for real-time luxury score visualization. 📈

---

## Acknowledgments
This project was completed as part of the MGSC 661: Multivariate Statistics course at McGill University, under the guidance of Professor Juan Camilo Serpa. 👨‍🏫
