import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pretty_confusion_matrix import pp_matrix_from_data

SEED = 12345

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

#hiperparametros
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
    verbose=True
)
model.fit(X_train, Y_train.ravel())

Y_predicted = model.predict(X_test)
columns = [ 'Malicioso', 'Legitimo']
pp_matrix_from_data(Y_test, Y_predicted, columns=columns)