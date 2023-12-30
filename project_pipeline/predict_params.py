import sys

import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def get_result(data, model_name, model):
    X_test, X_train, y_test, y_train = data
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    with open(predict_file, 'a') as f:
        f.write(f'{model_name}\n')
        f.write("-" * 88 + "\n")
        f.write(str(y_pred) + "\n")
        f.write(str(accuracy) + "\n")
        f.write("-" * 88 + "\n")


def main():
    try:
        models, data = joblib.load(models_and_data_file)
        for name, model in models.items():
            get_result(data, name, model)
    except Exception as e:
        print(f"Error occurs by loading data: {e}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 predict_model.py models_and_data_file predict_file")
        sys.exit(1)

    models_and_data_file = sys.argv[1]
    predict_file = sys.argv[2]
    main()
