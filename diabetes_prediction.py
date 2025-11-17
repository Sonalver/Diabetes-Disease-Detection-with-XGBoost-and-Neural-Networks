# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# LOAD DATA
df = pd.read_csv("C:/Users/sonal/PycharmProjects/PythonProject/resources/diabetes.csv")

# BASIC EXPLORATION
print(df.head())
print(df.info())
print(df.describe())

# DATA VISUALIZATION
plt.figure(figsize=(10,5))
sns.countplot(x='Outcome', data=df)
plt.title("Outcome Distribution")
plt.show()

sns.pairplot(df, hue='Outcome')
plt.show()

# DATA SPLIT & SCALING
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Proper split: 75% train – 25% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# BUILD ANN MODEL
ANN_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(400, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(400, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

ANN_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ANN_model.summary()

# TRAIN ANN
history = ANN_model.fit(X_train, y_train, epochs=60, batch_size=32, verbose=1)

# PLOT LOSS & ACCURACY
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(history.history['accuracy'])
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# PREDICTIONS ON TEST SET
y_pred_ann = ANN_model.predict(X_test)
y_pred_ann = (y_pred_ann > 0.5)

print("\nANN Classification Report:")
print(classification_report(y_test, y_pred_ann))

cm_ann = confusion_matrix(y_test, y_pred_ann)
sns.heatmap(cm_ann, annot=True, fmt='d')
plt.title("ANN Confusion Matrix")
plt.show()

# XGBOOST MODEL
xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

print("\nXGBoost Train Accuracy:", xgb_model.score(X_train, y_train))
print("XGBoost Test Accuracy:", xgb_model.score(X_test, y_test))

y_pred_xgb = xgb_model.predict(X_test)

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d')
plt.title("XGBoost Confusion Matrix")
plt.show()

# LOGISTIC REGRESSION
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print("\nLogistic Regression Test Accuracy:", log_reg.score(X_test, y_test))

y_pred_lr = log_reg.predict(X_test)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.show()
