import pandas as pd 
import mlflow 
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import joblib
import os

# GridSearchCV (teste toutes les combinaisons des hyperparametres)

def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df 

def train_model(): 
    # configuration MLflow
    mlflow.set_experiment("California_Housing_Price_Prediction")

    with mlflow.start_run():
        # chargement des données 
        df = load_data()
        X = df.drop('target', axis=1)
        Y = df['target']

        # split des données 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # entrainemetn du modèle

        params = { 
            'n_estimators' : 100,
            'max_depth' : 10,
            'random_state' : 42
        }

        model = RandomForestRegressor(**params)
        model.fit(X_train_scaled, Y_train)

        # predictions et metrics
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)

        # logs dans mlflow 
        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2_Score", r2)

        # sauvegarde du modele et le scaler
        mlflow.sklearn.log_model(
            sk_model=model,
            name="rf_model",
            registered_model_name="CaliforniaHousingRF"
        )

        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        mlflow.log_artifact('models/scaler.pkl')

        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        return model, scaler


if __name__ == "__main__":
    train_model()


