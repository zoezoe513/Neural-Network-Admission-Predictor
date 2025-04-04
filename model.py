from sklearn.neural_network import MLPClassifier

def train_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=123)
    model.fit(X_train, y_train)
    return model
