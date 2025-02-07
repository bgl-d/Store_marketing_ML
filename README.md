# Store-marketing
Store marketing campaign task from Kaggle*

Context- A superstore is planning for the year-end sale. They want to launch a new offer - gold membership, that gives a 20% discount on all purchases, for only $499 which is $999 on other days. It will be valid only for existing customers and the campaign through phone calls is currently being planned for them. The management feels that the best way to reduce the cost of the campaign is to make a predictive model which will classify customers who might purchase the offer.

Objective - The superstore wants to predict the likelihood of the customer giving a positive response and wants to identify the different factors which affect the customer's response. You need to analyze the data provided to identify these factors and then build a prediction model to predict the probability of a customer will give a positive response.

### Steps:
1. **Data preparation:** checking for duplicates, null values and outiers.
2. **Exploratory data analysis:** gaining insights from cleaned data.
3. **Data preprocessing:** encoding objective features like education and marital status and analysing correlations between varibales.
4. **Logit model implementation:**  data scailing, model building and evaluation of the parametres
5. **Random Forest model implementation:** model building and charateristics evaluation

### Data preparation
For the most part suggested store marketing data is complete and well cleaned. It doesn't have any duplicates and only 24 missing values (1% of the population) from income column. Most of the times the best replacment for income would be median. 

Multiple columns had outliers which were discovered using boxplot charts:

![Features boxplots](https://github.com/user-attachments/assets/45d1128a-cd1f-4e77-b26a-b8d5f1c9453e)


### EDA
After removing outliers and replacing missing data it became possible to evaluate each factor, create metrics and gain first insights.
Firstly I separated all features in two categories. One that showed each customer's personal features, e.g. income, number of kids and teens, marital status and education. 
![Previous campaign responses based on customer features](https://github.com/user-attachments/assets/b74b051a-369d-44fc-bba1-8363a896c1f7)

The second included customer engagement with the store (e.g. number of purcheses through different channels, number of purchases made with discount, complaince etc.).

![Previous campaign responses based on customer engagement](https://github.com/user-attachments/assets/8532a6c9-cbf7-4df6-b061-a80f03a16288)


### *Resources:
https://www.kaggle.com/datasets/ahsan81/superstore-marketing-campaign-dataset
