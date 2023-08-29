import numpy as np
import pandas as pd

import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

SEED = 12345
USE_PCA = True
USE_EXTERN_TOOLS = True

extern_attributes = [
    'domain_age',
    'page_rank',
    'has_DNS_record',
    'A_records',
    'AAAA_records',
    'avg_ttl',
    'avg_AAAA_ttl',
    'asn_count'
]

def get_model(model_name):

    if model_name == "mlp":
        functions = ('identity', 'logistic', 'tanh', 'relu')
        learning_rate = 0.075
        hidden_layer_sizes = (25, 40, 15)
        activation_function = functions[2]
        validation_percentage=.2
        max_epochs=500

        #modelo
        model = MLPClassifier(
            solver='sgd',
            random_state=SEED,
            learning_rate_init=learning_rate,
            learning_rate='adaptive',
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation_function,
            early_stopping=True,
            validation_fraction=validation_percentage,
            max_iter=max_epochs,
            verbose=False
        )
        return model
    
    if model_name == "bayes":
        return GaussianNB()
    
    if model_name == "dt":
        return DecisionTreeClassifier(random_state=SEED)
    
    if model_name == "knn":
        return KNeighborsClassifier(n_neighbors=3)

    if model_name == "rf":
        return RandomForestClassifier(random_state=SEED)

    if model_name == "svm":
        return SVC(random_state=SEED)


data = pd.read_csv("./datasets/dataset.csv", sep=',')

dataframe = data.copy()
dataframe = dataframe.astype({
    'is_ip': int,
    'url_has_at_char': int,
    'is_https': int,
    'url_redirection': int,
    'domain_has_dash_char': int,
    'url_has_https_token': int,
    'has_DNS_record': int,
    'has_iframe': int,
    'most_frequent_domain_in_anchor': int,
    'most_frequent_domain_in_link': int,
    'most_frequent_domain_in_source': int,
    'has_anchor_in_body': int,
    'phishing': int 
})

dataframe = dataframe.iloc[:, 1:]
features_dataframe = dataframe.iloc[:, 0:-1]

# remove atributos com variancia zero
print("\nVAR")
zero_variance_columns = features_dataframe.columns[features_dataframe.var() == 0]
print("ZERO VARIANCE COLUMNS = ")
print(zero_variance_columns)
features_dataframe = features_dataframe.drop(columns=zero_variance_columns)
features_dataframe = features_dataframe.drop(columns=['asn_count'])

# remove atributos com alta correlação
print("\nCORR")
sn.heatmap(
    features_dataframe.rename(
        columns={
            'url_length': 'A1',
            'is_ip': 'A2',
            'url_has_at_char': 'A3',
            'number_of_dots': 'A4',
            'domain_has_dash_char': 'A5',
            'url_has_https_token': 'A6',
            'url_redirection': 'A7',
            'is_https': 'A8',
            'page_rank': 'A9',
            'domain_age': 'A10',
            'has_DNS_record': 'A11',
            'A_records': 'A12',
            'AAAA_records': 'A13',
            'avg_ttl': 'A14',
            'avg_AAAA_ttl': 'A15',
            'asn_count': 'A16',
            'has_iframe': 'A17',
            'most_frequent_domain_in_anchor': 'A18',
            'most_frequent_domain_in_link': 'A19',
            'most_frequent_domain_in_source': 'A20',
            'common_page_rate': 'A21',
            'footer_common_page_rate': 'A22',
            'null_link_ratio': 'A23',
            'footer_null_link_ratio': 'A24',
            'has_anchor_in_body': 'A25',
        }
    ).corr()
)
plt.show()
features_dataframe = features_dataframe.drop(columns=['has_DNS_record'])

if (not USE_EXTERN_TOOLS):
    for column in extern_attributes:
        if (column in features_dataframe):
            features_dataframe = features_dataframe.drop(columns=[column])

X = np.array(features_dataframe.values)
Y = np.array(dataframe[['phishing']].values)

# substituindo valores inexistentes por 0
teste = np.isnan(X)
indexes = np.transpose((teste).nonzero())
for index in indexes:
    X[index[0], index[1]] = 0

#normalizacao
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#divisao entre treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, train_size=0.75, random_state=SEED)

# aplicação do PCA
if (USE_PCA):

    n_components = 15 if USE_EXTERN_TOOLS else 5
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(X_scaled)
    X_transformed_PCA = pca.transform(X_scaled)

models_names = [
    "mlp",
    "bayes",
    "knn",
    "dt",
    "rf",
    "svm"
]

x = []
y_acc = []
y_acc_err = []
y_prec = []
y_prec_err = []
y_f1 = []
y_f1_err = []
y_rec = []
y_rec_err = []

for model_name in models_names:

    model = get_model(model_name)

    metrics = ['accuracy', 'precision', 'f1', 'recall']

    if (model_name == "bayes"):
        scores = cross_validate(model, X_transformed_PCA, Y.ravel(), cv=5, scoring=metrics)
    else:
        scores = cross_validate(model, X_scaled, Y.ravel(), cv=5, scoring=metrics)

    accuracy = scores['test_accuracy']
    precision = scores['test_precision']
    f1 = scores['test_f1']
    recall = scores['test_recall']

    x.append(model_name.upper())

    print("\n>>>", model_name.upper())
    print("ACCURACY = {:.2f} +- {:.2f}".format(100*accuracy.mean(), 100*accuracy.std()))
    y_acc.append(accuracy.mean())
    y_acc_err.append(accuracy.std())
    print("PRECISION = {:.2f} +- {:.2f}".format(100*precision.mean(), 100*precision.std()))
    y_prec.append(precision.mean())
    y_prec_err.append(precision.std())
    print("RECALL = {:.2f} +- {:.2f}".format(100*recall.mean(), 100*recall.std()))
    y_rec.append(recall.mean())
    y_rec_err.append(recall.std())
    print("F1 = {:.2f} +- {:.2f}".format(100*f1.mean(), 100*f1.std()))
    y_f1.append(f1.mean())
    y_f1_err.append(f1.std())


plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

fig, axs = plt.subplots(2, 2)

axs[0, 0].errorbar(x, y_acc, y_acc_err, fmt='o', markerfacecolor='#4287f5', markeredgecolor='black', ecolor='black', capsize=10, markersize=10, markeredgewidth=2)
axs[0, 0].set_title('Acurácia')
axs[0, 0].grid()
axs[0, 0].set(ylim=[.89 if USE_EXTERN_TOOLS else .7, 1])

axs[0, 1].errorbar(x, y_prec, y_prec_err, fmt='o', markerfacecolor='#bf100a', markeredgecolor='black', ecolor='black', capsize=10, markersize=10, markeredgewidth=2)
axs[0, 1].set_title('Precisão')
axs[0, 1].grid()
axs[0, 1].set(ylim=[.89 if USE_EXTERN_TOOLS else .7, 1])

axs[1, 0].errorbar(x, y_rec, y_rec_err, fmt='o', markerfacecolor='#48b330', markeredgecolor='black', ecolor='black', capsize=10, markersize=10, markeredgewidth=2)
axs[1, 0].set_title('Recall')
axs[1, 0].grid()
axs[1, 0].set(ylim=[.89 if USE_EXTERN_TOOLS else .7, 1])

axs[1, 1].errorbar(x, y_f1, y_f1_err, fmt='o', markerfacecolor='#bfaa0a', markeredgecolor='black', ecolor='black', capsize=10, markersize=10, markeredgewidth=2)
axs[1, 1].set_title('F1')
axs[1, 1].grid()
axs[1, 1].set(ylim=[.89 if USE_EXTERN_TOOLS else .7, 1])

plt.show()