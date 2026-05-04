import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy import stats

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("diabetes.csv")

print("Original Shape:", df.shape)

# =========================
# 2. REDUCE DATA SIZE (IMPORTANT)
# =========================
df = df.sample(n=20000, random_state=42)

# Select important features
df = df[[
    "HighBP", "BMI", "Smoker", "Age",
    "GenHlth", "PhysHlth", "MentHlth",
    "Diabetes_012"
]]

print("After sampling + feature selection:", df.shape)

# =========================
# 3. PREPROCESSING
# =========================

# Convert target to binary
df["Diabetes_012"] = df["Diabetes_012"].apply(lambda x: 1 if x > 0 else 0)

# Remove outliers
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

print("After cleaning:", df.shape)

# =========================
# 4. SPLIT FEATURES & TARGET
# =========================
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# =========================
# 5. SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 6. PCA
# =========================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 7. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# =========================
# 8. TRAIN MODELS
# =========================

# SVM (Classification)
svm = SVC(probability=True)
svm.fit(X_train, y_train)

# Ridge (Risk score)
ridge = Ridge()
ridge.fit(X_train, y_train)

# KMeans (Clustering)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_pca)

# =========================
# 9. MODEL EVALUATION
# =========================
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# =========================
# 10. SAVE MODELS
# =========================
pickle.dump(svm, open("svm.pkl", "wb"))
pickle.dump(ridge, open("ridge.pkl", "wb"))
pickle.dump(kmeans, open("kmeans.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pca, open("pca.pkl", "wb"))

print("All models saved successfully!")