# Store-marketing
**Store marketing campaign task from Kaggle***



This project is based on superstore's marketing campaign with data from Kaggle. Its objective is to showcase my skills in data science by building model and predicting the likelihood of customers responding positively to a year-end sale campaign for a superstore.

---

  **Context:** A superstore is planning for the year-end sale. They want to launch a new offer - gold membership, that gives a 20% discount on all purchases, for only $499 which is $999 on other days. It will be valid only for existing customers and the campaign through phone calls is currently being planned for them. The management feels that the best way to reduce the cost of the campaign is to make a predictive model which will classify customers who might purchase the offer.

  **Objective:** The superstore wants to predict the likelihood of the customer giving a positive response and wants to identify the different factors which affect the customer's response. You need to analyze the data provided to identify these factors and then build a prediction model to predict the probability of a customer will give a positive response.

---

### Steps:

**1. Data cleaning:** Handle missing values, duplicates, and outliers.

**2. Data transforamtion:** Encode categorical variables and transform date features.

**3. Exploratory data analysis:** Analyze feature correlations and distributions.

**4. Data reduction:** Perform feature selection, scaling, and sampling to balance the dataset.

**5. Model builidng:** 

 * Implement a Logistic Regression model.

 * Implement a Random Forest model.  

**6. Model Evaluation:** Evaluate models using metrics like AUC, confusion matrix, and classification reports.


### Code Implementation

**1. Data Cleaning (cleaning):**

* Check for duplicates and missing values.
* Handle missing values by imputing the median for the Income column.
* Visualize outliers using boxplots.

**2. Data Transformation (transformation):**

* Convert enrollment dates to integer values.
* Perform ordinal encoding for categorical variables.

**3. Exploratory Data Analysis (eda):**

* Generate a correlation matrix to understand feature relationships.
* Visualize correlations using a heatmap.

**4. Data Reduction (reduction):**

* Drop less significant features based on p-values.
* Scale features using StandardScaler.
* Balance the dataset using SMOTE (Synthetic Minority Over-sampling Technique).

**5. Logistic Regression Model (logit_model):**

* Split data into training and testing sets.
* Fit a logistic regression model using statsmodels.
* Evaluate the model using a confusion matrix, ROC curve, and AUC.**

**6. Random Forest Model (random_forest):**

* Split data into training and testing sets.
* Fit a Random Forest classifier using scikit-learn.
* Evaluate the model using a confusion matrix and AUC.**

### Results







### *Resources:
https://www.kaggle.com/datasets/ahsan81/superstore-marketing-campaign-dataset

## **Model evaluation:
