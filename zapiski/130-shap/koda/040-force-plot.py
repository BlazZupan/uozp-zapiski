import shap
import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Naložimo podatke
heart = fetch_openml(name='heart-statlog', version=1, as_frame=True)
X = heart.data
y = (heart.target == 'present').astype(int)  # Convert 'present'/'absent' to 1/0

# Print actual column names to see what we're working with
print("Actual column names:", X.columns.tolist())

# Create shorter feature names based on actual column names
feature_names = {
    'age': 'Age',
    'sex': 'Sex',
    'chest': 'Chest',
    'resting_blood_pressure': 'BP',
    'serum_cholestoral': 'Chol',
    'fasting_blood_sugar': 'Sugar',
    'resting_electrocardiographic_results': 'ECG',
    'maximum_heart_rate_achieved': 'HR',
    'exercise_induced_angina': 'Angina',
    'oldpeak': 'ST',
    'slope': 'Slope',
    'number_of_major_vessels': 'Vessels',
    'thal': 'Thal'
}

# Naučimo model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# SHAP razlaga
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Poiščemo prvi primer, kjer je napovedan razred 1
predictions = model.predict(X_test)
index = list(predictions).index(1)  # prvi primer z razredom 1

# Force plot za ta primer
force_plot = shap.force_plot(
    base_value=shap_values.base_values[index],
    shap_values=shap_values.values[index],
    features=X_test.iloc[index],
    feature_names=[feature_names[col] for col in X_test.columns],
    matplotlib=True,
    plot_cmap="RdBu",
    text_rotation=0,
    contribution_threshold=0.05,
    link="identity",
    show=False
)
plt.gcf().set_size_inches(8, 4)  # Set figure size after plot is created
plt.rcParams.update({'font.size': 12})  # Increase font size
plt.tight_layout()
plt.savefig('force-plot.svg', bbox_inches='tight', pad_inches=0.1, dpi=300)  # Save before showing
plt.show()  # Show after saving