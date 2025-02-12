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


def cleaning(df):
    # check for duplicates
    print(df.duplicated().sum())

    # null values
    print(df.isnull().sum())
    df['Income'] = df['Income'].fillna(df['Income'].median())

    # plot boxplot charts
    fig, ax = plt.subplots(ncols=5, nrows=4, figsize=(16, 15))
    boxplots = df.drop(columns='Dt_Customer')
    col = 0
    for i in range(4):
        for j in range(5):
            sns.boxplot(boxplots[boxplots.columns[col]], ax=ax[i][j])
            ax[i][j].set_title(boxplots.columns[col])
            col = col + 1


    df.drop(df[df['Income'] > 300000].index, inplace=True)

    # sort by date
    df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer).dt.to_period('M')
    df.Dt_Customer.sort_values()
    return df


def transformation(df):
    # enrollment dates to int
    sorted_dt = sorted([str(val) for val in df.Dt_Customer.unique()])
    df['Dt_Customer'] = df.Dt_Customer.apply(lambda val: sorted_dt.index(str(val)))
    df.drop('Id', axis=1, inplace=True)

    # ordinal encoding
    object_cols = df.select_dtypes(include=['object']).columns
    ordinal_encoder = OrdinalEncoder()
    df[object_cols] = ordinal_encoder.fit_transform(dataset[object_cols])
    return df


def eda(df):
    # correlation matrix
    fig3, ax3 = plt.subplots(figsize=(16, 9))
    x_labels = df.columns
    corr_m = df.corr()
    sns.heatmap(corr_m, cmap="YlGnBu", annot=True, annot_kws={"size": 8}, ax=ax3)
    ax3.set_xticklabels(x_labels, rotation=30, ha='right')
    print(df.describe())


def reduction(df):
    # scaling
    X = df.drop(['Response', 'Year_Birth', 'Kidhome', 'MntFruits', 'MntSweetProducts',
                 'MntFishProducts', 'MntGoldProds', 'Complain', 'Income', 'Marital_Status'], axis=1)  # drop features with p < 0.05
    y = df['Response']
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)

    # samppling
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    return X_resampled,y_resampled


def logit_model(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        stratify=y, random_state=0, train_size=.8)

    # logit model
    m_logit = sm.Logit(y_train, X_train).fit()
    predict_l = m_logit.predict(X_test)
    predictions = (predict_l[:] > 0.6).astype(int)

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

    # AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(auc)


if __name__ == '__main__':
    # terminal display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 400)
    np.set_printoptions(linewidth=400)

    # load data
    dataset = pd.read_csv('../Store marketing/superstore_data.csv')

    # filling missing data, removing outliers
    dataset = cleaning(dataset)

    # data transformation
    dataset = transformation(dataset)

    # exploratory data analysis
    eda(dataset)

    # data reduction
    x, y = reduction(dataset)

    # logit model building and evaluation
    # logit_model(x, y)

    # Random Forest model
    random_forest(x, y)
