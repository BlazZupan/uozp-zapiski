import shap
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Naložimo podatke (Cleveland Heart Disease)
heart = fetch_openml(name='heart-statlog', version=1, as_frame=True)
X = heart.data
y = (heart.target == 'present').astype(int)  # Convert 'present'/'absent' to 1/0

# Naučimo model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# SHAP razlage
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Prikaz beeswarm grafa z več prostora za oznake
plt.figure(figsize=(10, 8))  # Increase figure size
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.savefig('beeswarm.svg', bbox_inches='tight', pad_inches=0.5)  # Add padding when saving
plt.show()
