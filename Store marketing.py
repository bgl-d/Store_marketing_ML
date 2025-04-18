import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px


def cleaning(df):
    # Sort by enrollment date
    df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer).dt.to_period('M')
    df.Dt_Customer.sort_values()

    # Duplicates
    print('Number of duplicates: ', df.duplicated().sum(), '\n')

    # Missing values
    print('Missing values: \n', df[df.isnull().any(axis=1)], '\n')
    df['Income'] = df['Income'].fillna(df['Income'].median())

    # Marital status values fix
    df.drop(df.loc[df['Marital_Status'] == 'YOLO'].index, inplace=True)
    df.drop(df.loc[df['Marital_Status'] == 'Absurd'].index, inplace=True)
    df.drop(df.loc[df['Marital_Status'] == 'Alone'].index, inplace=True)

    # Age and age groups instead of birth year
    df['Age'] = 2025 - df['Year_Birth']
    bins = [0, 25, 35, 45, 55, 65, 130]
    labels = ['<25', '25–34', '35–44', '45–54', '55–64', '65+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df = df.drop(columns=['Year_Birth'])

    # Spending groups
    df['Total_Spending'] = (df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] +
                            df['MntSweetProducts'] + df['MntGoldProds'])
    bins = [0, 250, 500, 1000, 1500, 2000, 3000]
    labels = ['<250', '250–500', '500–1000', '1000–1500', '1500–2000', '2000+']
    df['SpendingGroup'] = pd.cut(df['Total_Spending'], bins=bins, labels=labels, right=False)
    return df


def eda(df):
    # EDA
    print(df.describe())

    # Extreme values
    for i in df.columns:
          fig = px.histogram(df, x=i, histnorm='percent')
          fig.update_layout(xaxis_title=i, bargroupgap=0.1)
          # fig.show()

    # Exclude extreme values
    extreme_values = {'Variable': ['NumWebVisitsMonth', 'NumWebPurchases', 'NumDealsPurchases', 'Age',
                                   'NumCatalogPurchases', 'MntSweetProducts', 'MntGoldProds', 'Income'],
                      'Separating_point': [10, 15, 14, 100, 15, 200, 250, 140000]}

    for i in range(len(extreme_values['Variable'])):
        if extreme_values['Separating_point'][i] > 0:
            df = df.drop(df[df[extreme_values['Variable'][i]] > extreme_values['Separating_point'][i]].index)
        else:
            df = df.drop(df[df[extreme_values['Variable'][i]] < abs(extreme_values['Separating_point'][i])].index)
    return df


def data_analysis_on_customer_segments(df):
    # Customer profile
    dimensions = ['Kidhome', 'Teenhome', 'Education', 'Marital_Status', 'Complain', 'AgeGroup', 'SpendingGroup']
    for i in dimensions:
        df.groupby('Response')[i].value_counts().unstack().T.plot(kind='bar', rot=0)
        plt.xlabel("Parameter")
        plt.ylabel("Count")
        plt.title(i)
        plt.legend(["Negative", "Positive"], title="Response")

    # Average deals through different channels
    channels = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    fig0, ax0 = plt.subplots(figsize=(16, 9))
    df.groupby('Response')[channels].mean().T.plot(kind='bar', rot=0, ax=ax0)
    ax0.set_ylabel("Mean Value")
    ax0.set_title('Channels')
    ax0.legend(["Negative", "Positive"], title="Response")

    # Mean income within different response
    income_means = df.groupby('Response')['Income'].mean().to_frame().T
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    income_means.plot(kind='bar', rot=0, ax=ax1)
    ax1.set_ylabel("Mean Value")
    ax1.set_title("Income")
    ax1.legend(["Negative", "Positive"], title="Response")

    # Mean shopping recency within different response
    mean_recency = df.groupby('Response')['Recency'].mean().to_frame().T
    fig2, ax2 = plt.subplots(figsize=(16, 9))
    mean_recency.plot(kind='bar', rot=0, ax=ax2)
    ax2.set_ylabel("Mean Value")
    ax2.set_title('Recency')
    ax2.legend(["Negative", "Positive"], title="Response")

    # Customers by spendings
    spending_per_age_group = df.groupby('AgeGroup', observed=True)['Total_Spending'].mean().to_frame().T
    fig5, ax5 = plt.subplots(figsize=(16, 9))
    spending_per_age_group.plot(kind='bar', rot=0, ax=ax5)
    ax5.set_ylabel("Mean Value")


def transformation(df):
    # enrollment dates to int
    sorted_dt = sorted([str(val) for val in df.Dt_Customer.unique()])
    df['Dt_Customer'] = df.Dt_Customer.apply(lambda val: sorted_dt.index(str(val)))

    # ordinal encoding
    object_cols = df.select_dtypes(include=['object', 'category']).columns
    ordinal_encoder = OrdinalEncoder()
    df[object_cols] = ordinal_encoder.fit_transform(df[object_cols])

    # correlation matrix
    fig3, ax3 = plt.subplots(figsize=(16, 9))
    corr_m = df.corr()
    x_labels = corr_m.columns
    sns.heatmap(corr_m, cmap="YlGnBu", annot=True, annot_kws={"size": 8}, ax=ax3)
    ax3.set_xticklabels(x_labels, rotation=30, ha='right')
    return df


def reduction(df):
    # Scaling
    X = df.drop(columns='Response')
    y = df['Response']
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)

    # Sampling
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    return X_resampled, y_resampled


def support_vector_reg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        stratify=y, random_state=0, train_size=.8)
    svr = SVR(kernel = 'rbf')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    y_pred = (y_pred >= 0.5).astype(int)
    print(metrics.classification_report(y_test, y_pred, target_names=['reject', 'accept']))

    # confusion matrix
    fig, ax = plt.subplots()
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('True')
    plt.xlabel('Prediction')
    print(cnf_matrix)
    # plt.show()

    # AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    print('AUC = ', auc)


def k_neighdbors(X, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        stratify=y, random_state=0, train_size=.8)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, target_names=['reject', 'accept']))

    # confusion matrix
    fig, ax = plt.subplots()
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('True')
    plt.xlabel('Prediction')
    print(cnf_matrix)
    # plt.show()

    # AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    print('AUC = ', auc)


def random_forest(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                            stratify=y, random_state=0, train_size=.8)

    # model fitting
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # model evaluating
    print(metrics.classification_report(y_test, y_pred, target_names=['reject', 'accept']))

    # confusion matrix
    fig, ax = plt.subplots()
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('True')
    plt.xlabel('Prediction')
    print(cnf_matrix)
    # plt.show()

    # AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    print('AUC = ', auc)


if __name__ == '__main__':
    # terminal display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 400)
    np.set_printoptions(linewidth=400)

    # load data
    dataset = pd.read_csv('../Store marketing/superstore_data.csv')

    # filling missing data, removing outliers
    clean_dataset = cleaning(dataset)

    # exploratory data analysis
    clean_dataset = eda(clean_dataset)

    # specific customer segments analysis
    data_analysis_on_customer_segments(clean_dataset)

    # data transformation
    clean_dataset = transformation(clean_dataset)

    # data reduction
    x, y = reduction(clean_dataset)

    # support vector regression
    support_vector_reg(x, y)

    # K-Nearest Neighbors
    k_neighdbors(x, y)

    # Random Forest model
    random_forest(x, y)