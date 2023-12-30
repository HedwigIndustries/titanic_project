import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def split_data(df, predict_param):
    X = df.drop(columns=[predict_param])
    y = df[predict_param]
    X_train, _X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return _X_test, X_train, y_test, y_train


def train_model(df, predict_param):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC()
    }
    X_test, X_train, y_test, y_train = split_data(df, predict_param)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # models_acc = {}

    # models_predicted_params = {}
    fitted_models = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        fitted_models[model_name]= model
        # y_pred = model.predict(X_test_scaled)
        # accuracy = accuracy_score(y_test, y_pred)
        # models_acc[model_name] = accuracy
        # models_predicted_params[model_name] = y_pred
    joblib.dump((fitted_models, [X_test, X_train, y_test, y_train]), output_file_name)


def main():
    df = pd.read_csv(prepared_data)
    train_model(df=df, predict_param="Pclass")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: python3 train_model.py prepared_data output')
        sys.exit(1)
    prepared_data = sys.argv[1]
    output_file_name = sys.argv[2]
    main()
