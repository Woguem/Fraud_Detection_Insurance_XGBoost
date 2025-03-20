import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Charger les données
#data = pd.read_csv('assurance_data.csv')

import pandas as pd
import numpy as np

# Générer un dataset factice
np.random.seed(42)
data_size = 20000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, data_size),
    'income': np.random.randint(20000, 100000, data_size),
    'policy_amount': np.random.randint(5000, 50000, data_size),
    'num_claims': np.random.randint(0, 5, data_size),
    'fraudulent': np.random.choice([0, 1], size=data_size, p=[0.7, 0.3])  # 40% de fraude
})

# Sauvegarder en CSV
data.to_csv('assurance_data.csv', index=False)

print("Dataset factice généré : assurance_data.csv")


# Prétraitement
X = data.drop('fraudulent', axis=1)  # Variables explicatives
y = data['fraudulent']  # Variable cible

# Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de classification
scale_pos_weight = len(y_train) / (2 * sum(y_train == 1))
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, learning_rate=0.6)
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))










