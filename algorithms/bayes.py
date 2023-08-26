import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.decomposition import PCA

SEED = 12345
usePca = True

data = pd.read_csv("../datasets/dataset.csv", sep=',')

new_data = data.copy()
new_data = new_data.astype({
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
    'legitimate': int 
})

X = np.array(new_data.iloc[:, 1:-1].values)
Y = np.array(new_data[['legitimate']].values)

teste = np.isnan(X)
print(teste)
indexes = np.transpose((teste).nonzero())
print(indexes)
for index in indexes:
    X[index[0], index[1]] = 0


#normalizacao
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#divisao entre treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, train_size=0.75, random_state=SEED)

# o naive bayes categorico do numpy nao aceita valores negativos
# X_train = -1 * X_train
# X_test = -1 * X_test

# discretização
# n_decimals = 1
# X_train = np.around(X_train, decimals=n_decimals)
# X_test = np.around(X_test, decimals=n_decimals)

if (usePca):

    # aplicação do PCA
    pca = PCA(n_components=15, random_state=SEED)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

model = GaussianNB()
model.fit(X_train, Y_train.ravel())

Y_predicted = model.predict(X_test)
columns = [ 'Malicioso', 'Legitimo']
pp_matrix_from_data(Y_test, Y_predicted, columns=columns)