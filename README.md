# Sentiment Analysis with Machine Learning

## Overview
This project focuses on sentiment analysis of user-generated messages collected from social media platforms. The goal is to classify sentiments (positive/negative) from text data using supervised and unsupervised machine learning techniques. This project was completed as part of an academic course in Machine Learning.

## Dataset
The dataset consists of social media messages, including metadata such as message ID, creation date, user verification status, and platform. The dataset is divided into training and validation sets, following the **holdout method** with an 80-20 split.

### Key Features:
- **Text data**: The core feature for sentiment analysis.
- **Metadata**: Includes user attributes like follower count, message history, and platform.
- **Label**: Sentiment of the message (positive/negative).

## Methodology
### Part 1: Dataset Preparation
1. **Exploratory Data Analysis**:
   - Visualized distributions of key features.
   - Investigated relationships between features and the target variable.
   - Addressed class imbalance through data transformations.
2. **Preprocessing**:
   - Performed feature selection based on significance.
   - Cleaned and normalized data to improve model performance.
   - Addressed missing values using imputation methods.

### Part 2: Model Training and Evaluation
1. **Models Implemented**:
   - **Decision Trees**: Tuned depth, split criteria, and other hyperparameters using random search.
   - **Artificial Neural Networks (ANN)**: Adjusted neuron count, activation functions, and solvers.
   - **Support Vector Machines (SVM)**: Optimized C parameter for best classification results.
   - **K-Medoids Clustering**: Conducted unsupervised clustering to explore patterns without labels.

2. **Validation**:
   - Employed AUC-ROC as the primary evaluation metric for classification tasks.
   - Validated results across different folds to ensure model reliability.

3. **Key Findings**:
   - **Best Performing Model**: ANN achieved the highest validation AUC-ROC score (~0.997), indicating superior accuracy.
   - Feature importance analysis highlighted user follower count and message frequency as critical determinants of sentiment.

4. **Model Improvement**:
   - Experimented with dimensionality reduction (PCA) and hyperparameter tuning.
   - Analyzed the impact of removing incomplete data versus imputation.

## Results
- The ANN model outperformed other approaches with minimal overfitting.
- Decision Trees provided interpretability, showing clear decision boundaries for sentiment classification.
- SVM and clustering offered complementary insights into data structure and feature relevance.


## How to Use
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
4. Train the models:
   ```bash
   python train_model.py --model [decision_tree/ann/svm]
   ```
5. Evaluate results:
   ```bash
   python evaluate.py
   ```

## Authors
- **Maya Adar**  
- **Tamar Yosef-Hay**

---
