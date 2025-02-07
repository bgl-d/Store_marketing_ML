import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import classification_report


def preprocessing(df):
    print(df.shape)
    print(df.dtypes)
    # date column to datetime
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

    # duplicates
    df.drop_duplicates()

    # income outliers
    df.drop(df[df.Income > 120000].index, inplace=True)

    # missing values
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]
    df[cols_with_missing[0]] = df[cols_with_missing[0]].fillna(df[cols_with_missing[0]].median())
    df.drop(df[df['Marital_Status'] == 'YOLO'].index, inplace=True)
    df.drop(df[df['Marital_Status'] == 'Absurd'].index, inplace=True)
    return df


def eda(df):
    # previous campaign response count
    customer_features = ['Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'Complain']
    fig1 = plt.figure(figsize=(9, 12))
    for i in range(len(customer_features)):
        ax1 = plt.subplot(5, 2, i + 1)
        sns.countplot(data=df, x=customer_features[i], hue='Response', ax=ax1, stat='count').set_title(customer_features[i])
        ax1.set(xlabel='')
    fig1.tight_layout()

    # correlation matrix
    fig2, ax2 = plt.subplots(figsize=(16, 9))
    df_heatmap = df.drop(columns=['Id', 'Dt_Customer', 'Education', 'Marital_Status'])
    x_labels = df_heatmap.columns
    corr_m = df_heatmap.corr()
    sns.heatmap(corr_m, cmap="YlGnBu", annot=True, annot_kws={"size": 8}, ax=ax2)
    ax2.set_xticklabels(x_labels, rotation=30, ha='right')

    # customer engagement
    engagement_features = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    fig3 = plt.figure(figsize=(9, 12))
    for i in range(len(engagement_features)):
        ax3 = plt.subplot(5, 2, i+1)
        sns.countplot(data=df, x=engagement_features[i], hue='Response', ax=ax3).set_title(engagement_features[i])
        ax3.set(xlabel='')
    fig3.tight_layout()

    # save pictures
    # fig2.savefig('heatmap.png')

    # previous campaign conversion rate
    rate = df['Response'].sum() * 100 / df.shape[0]
    print('Previous campaign conversion rate = ', rate)

    # quantitative values eda
    print(df.describe())


def encoding(df):
    # ordinal encoding
    object_cols = df.select_dtypes(include=['object']).columns
    ordinal_encoder = OrdinalEncoder()
    df[object_cols] = ordinal_encoder.fit_transform(dataset[object_cols])
    print(df.dtypes)
    return df


def model_building(df):
    # train, test data split
    x = df.drop(columns=['Response', 'Dt_Customer'])
    y = df[['Response']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=.8)

    # logit model
    m_logit = sm.Logit(y_train, X_train).fit()
    predict_l = m_logit.predict(X_test)
    predict_l = (predict_l[:] > 0.5).astype(int)

    # model's summary
    print(m_logit.summary())
    # print(classification_report(y_test, predict_l))

    # confusion matrix
    cnf_matrix_l = metrics.confusion_matrix(y_test, predict_l)

    # create heatmap
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix_l), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    # plt.show()

    return predict_l


#def white_test(pred_l, pred_p):
    # White's test
    # X_train['Const'] = 1
    # white_test = d.het_breuschpagan(m_probit.resid_generalized, exog_het=X_train[['NumCatalogPurchases','Const']])
    # labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']
    # print(white_test)


if __name__ == '__main__':
    # terminal display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 400)
    np.set_printoptions(linewidth=400)

    # load data
    dataset = pd.read_csv('../Store marketing/superstore_data.csv')

    # deal with missing data, duplicates
    dataset = preprocessing(dataset)

    # exploratory data analysis
    eda(dataset)

    # encoding object variables
    dataset = encoding(dataset)

    # logit and probit model building and evaluation
    pred_l, pred_p = model_building(dataset)