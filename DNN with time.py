#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:44:32 2024

@author: snlab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import time

# Load dataset
data = pd.read_csv('/home/snlab/EITO/Dataset/task_offloading_dataset.csv')
print(f'Dataset shape: {data.shape}')

# Preprocessing
# Encode the Offloading_Decision
le = LabelEncoder()
data['Offloading_Decision'] = le.fit_transform(data['Offloading_Decision'])

# Define features and target variable
X = data.drop(columns=['Offloading_Decision'])
y = data['Offloading_Decision']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create DNN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(32, activation='relu'))  # Second hidden layer
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))  # Second hidden layer
model.add(Dropout(0.3))
model.add(Dense(8, activation='relu'))  # Second hidden layer
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))  # Second hidden layer
model.add(Dropout(0.3))
model.add(Dense(2, activation='relu'))  # Second hidden layer
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Print the number of neurons in each layer
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'units'):
        print(f'Layer {i + 1}: {layer.units} neurons')
    else:
        print(f'Layer {i + 1}: {layer.name} (not applicable for this layer)')

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the learning rate
learning_rate = model.optimizer.learning_rate.numpy()  # Access the learning rate
print(f'Learning Rate: {learning_rate:.6f}')

# Train the model and measure training time
start_time = time.time()  # Record start time
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
elapsed_time_ms = (time.time() - start_time) * 1000  # Calculate elapsed time in milliseconds
print(f'Training Time: {elapsed_time_ms:.2f} ms')

# Retrieve training and validation accuracy
train_accuracy = history.history['accuracy'][-1]  # Last epoch's training accuracy
val_accuracy = history.history['val_accuracy'][-1]  # Last epoch's validation accuracy

# Evaluate the model on the test set
loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Train Accuracy: {train_accuracy*100:.2f}%')
print(f'Validation Accuracy: {val_accuracy*100:.2f}%')
print(f'Test Accuracy: {test_accuracy*100:.2f}%')

# Predictions on test data
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary decisions

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print model summary
model.summary()

# Calculate model size
total_params = model.count_params()
model_size_bytes = total_params * 4  # Assuming 4 bytes per parameter
print(f'Total number of parameters: {total_params}')
print(f'Model size (approx): {model_size_bytes / (1024 * 1024):.2f} MB')  # Convert to MB

# ROC Curve
y_pred_prob = model.predict(X_test)  # Predictions are probabilities for the positive class (1)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)  # Compute FPR, TPR for different thresholds
auc = roc_auc_score(y_test, y_pred_prob)  # Compute AUC score

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line (random guessing)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
