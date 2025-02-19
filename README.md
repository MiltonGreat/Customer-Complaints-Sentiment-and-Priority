# Customer-Complaints-Sentiment-and-Priority Dataset Analysis

## Overview

This repository contains the analysis and machine learning models built on the Customer Complaints Sentiment and Priority Dataset. The dataset is derived from consumer complaints in the financial sector and includes both sentiment analysis and priority classification of complaints. The project leverages Natural Language Processing (NLP) techniques and machine learning algorithms to automate customer complaint categorization and ensure timely responses to high-priority issues.

By utilizing NLP and classification models, this project aims to help financial institutions improve their customer service operations, automate complaint triage, and enhance user satisfaction by addressing priority complaints promptly.

### Project Overview

In this project, we aim to:
- Clean and preprocess customer complaints.
- Perform sentiment analysis to classify complaints as positive or negative.
- Classify complaint priority based on urgency using a variety of machine learning models.
- Identify key themes and patterns in the complaints through topic modeling.
- Visualize important features for priority classification.

### Dataset Description

The dataset contains the following columns:

- Consumer_complaint: The full complaint description from the customer.
- Sentiment: The sentiment label for each complaint (binary: 0 = Negative, 1 = Positive).
- Priority: The priority level of the complaint (e.g., High, Medium, Low).
- Product: The product related to the complaint (e.g., Credit Card, Mortgage).

### Data Preprocessing

The following preprocessing steps were performed:

- Text Cleaning: Removed special characters, numbers, and extra spaces from the complaint texts.
- Resampling for Imbalanced Classes: Used random oversampling to balance the sentiment classes (positive and negative).
- Stopwords Removal: Removed common stopwords to improve model performance.
- Feature Extraction: Used TfidfVectorizer to extract features from the complaint texts.

### Machine Learning Models

#### Sentiment Classification (Random Forest)
- Data: Consumer complaints labeled as Positive (1) or Negative (0).
- Model: Random Forest Classifier, optimized using GridSearchCV.
- Results: The model achieved an accuracy of 95.19%.

#### Priority Classification (Random Forest)
- Data: Consumer complaints labeled with different priority levels.
- Model: Random Forest Classifier with SMOTE for handling class imbalance.
- Results: The model achieved an accuracy of 95.19%.

### Topic Modelling Analysis

To gain deeper insights into customer complaints, we performed Latent Dirichlet Allocation (LDA) topic modeling. The analysis revealed the following topics:

- Topic 0: Accounts, payments, and loan-related issues.
- Topic 1: Complaints related to loan payments and mortgages.
- Topic 2: Credit-related issues, including late payments and payment processing.
- Topic 3: Credit reporting and debt collection, with a focus on reporting accuracy and dispute resolution.
- Topic 4: Issues with bank accounts, money transfers, and funds management.

These topics highlight the key areas where consumers are facing challenges, particularly in financial services like credit cards, mortgages, and payments.

### Results and Evaluation

- Sentiment Classification (Random Forest): Accuracy = 95.19%. The model performs well in predicting sentiment based on complaint text.
- Priority Classification (Random Forest): Accuracy = 95.19%. The model was able to predict complaint priority with high precision.

### Model Visualizations

Top 20 Important Features: A bar chart showing the most important words/features for priority classification, derived from the Random Forest model.

![screenshot-localhost_8890-2025 02 02-15_46_06](https://github.com/user-attachments/assets/63bc2c9d-a27a-4740-93df-746e6225bbc9)

### Source

Dataset: [Customer Complaints Sentiment and Priority Dataset on Kaggle](https://www.kaggle.com/datasets/xjoury/customer-complaints-sentiment-and-priority-dataset)
