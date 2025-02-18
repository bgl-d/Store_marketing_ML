import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
    return df


def eda(df):
    df = df.drop(columns=['Id'])
    print(df.describe(), '\n')

    # Histograms
    hist_df = df.drop(columns=['Dt_Customer'])
    for i in hist_df.columns:
          fig = px.histogram(hist_df, x=i, histnorm='percent')
          fig.update_layout(xaxis_title=i, bargroupgap=0.1)
          fig.show()

    # Exclude extreme values
    extreme_values = {'Variable': ['NumWebVisitsMonth', 'NumWebPurchases', 'NumDealsPurchases', 'Year_Birth',
                                   'NumCatalogPurchases', 'MntSweetProducts', 'MntGoldProds', 'Income'],
                      'Separating_point': [10, 15, 14, -1920, 15, 200, 250, 140000]}

    for i in range(len(extreme_values['Variable'])):
        if extreme_values['Separating_point'][i] > 0:
            df = df.drop(df[df[extreme_values['Variable'][i]] > extreme_values['Separating_point'][i]].index)
        else:
            df = df.drop(df[df[extreme_values['Variable'][i]] < abs(extreme_values['Separating_point'][i])].index)
    return df


def transformation(df):
    # enrollment dates to int
    sorted_dt = sorted([str(val) for val in df.Dt_Customer.unique()])
    df['Dt_Customer'] = df.Dt_Customer.apply(lambda val: sorted_dt.index(str(val)))

    # ordinal encoding
    object_cols = df.select_dtypes(include=['object']).columns
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
    return X_resampled,y_resampled


def logit_model(x, y):
    # excluding features with p-value > 0.05 after model fitting
    x = x.drop(columns=['NumWebPurchases', 'Year_Birth', 'Kidhome', 'MntFruits',
                        'MntGoldProds', 'Complain', 'Marital_Status', 'MntSweetProducts'])

    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        stratify=y, random_state=0, train_size=.8)

    # logit model
    m_logit = sm.Logit(y_train, X_train).fit()
    predict_l = m_logit.predict(X_test)
    predictions = (predict_l[:] > 0.5).astype(int)

    # model's summary
    print(m_logit.summary())
    print(metrics.classification_report(y_test, predictions, target_names=['reject', 'accept']))

    # confusion matrix
    cnf_matrix = metrics.confusion_matrix(y_test, predictions)

    # confusion matrix heatmap
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    fig.tight_layout()

    # ROC curve & AUC
    fig1, ax1 = plt.subplots()
    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
    auc = metrics.roc_auc_score(y_test, predictions)
    ax1.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    print(auc)

    # vif
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
    print(vif_data)


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
    plt.show()

    # AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(auc)

    # features
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance)


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

    # data transformation
    clean_dataset = transformation(clean_dataset)

    # data reduction
    x, y = reduction(clean_dataset)

    # logit model building and evaluation
    logit_model(x, y)

    # Random Forest model
    random_forest(x, y)
