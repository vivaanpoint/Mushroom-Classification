import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# LOAD AND SIMULATE DATA
np.random.seed(42)
n_samples = 1000

data = {
    'class': np.random.choice(['edible', 'poisonous'], size=n_samples, p=[0.52, 0.48]),
    'cap-shape': np.random.choice(['convex', 'flat', 'sunken', 'bell'], size=n_samples),
    'cap-color': np.random.choice(['brown', 'yellow', 'white', 'gray', 'red'], size=n_samples),
    'bruises': np.random.choice(['bruises', 'no'], size=n_samples),
    'odor': np.random.choice(['almond', 'anise', 'foul', 'musty', 'none'], size=n_samples, p=[0.1, 0.1, 0.3, 0.1, 0.4]),
    'gill-size': np.random.choice(['broad', 'narrow'], size=n_samples),
    'spore-print-color': np.random.choice(['black', 'brown', 'white', 'chocolate'], size=n_samples),
    'population': np.random.choice(['abundant', 'clustered', 'numerous', 'scattered', 'several'], size=n_samples)
}

df = pd.DataFrame(data)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# PREPROCESSING & ENCODING
mappings = {}
le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])
    mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\nEncoding Example (Class):", mappings['class'])
print("Encoding Example (Odor):", mappings['odor'])

# Split features (X) and target label (y)
X = df.drop('class', axis=1)
y = df['class']

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAIN THE MODEL
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\nModel training complete.")

# EVALUATION
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['edible', 'poisonous']))

# VISUALIZING FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis")
plt.title("Which Mushroom Features Matter Most?")
plt.xlabel("Relative Importance Score")
plt.ylabel("Mushroom Feature")
plt.tight_layout()
plt.show()