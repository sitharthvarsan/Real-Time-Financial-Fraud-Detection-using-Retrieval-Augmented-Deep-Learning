# train_models.py

import torch
import torch.nn as nn
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_processing import load_and_preprocess_data

X_resampled, y_resampled, X_test, y_test = load_and_preprocess_data()

# --- Train and Save XGBoost Model ---
print("Training XGBoost Model...")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb_model.fit(X_resampled, y_resampled)
y_pred_xgb = xgb_model.predict(X_test)
joblib.dump(xgb_model, 'xgb_model.joblib')

print("\nXGBoost Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.show()

# --- Train and Save PyTorch NN Model ---
class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super(FraudNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

input_dim = X_test.shape[1]
nn_model = FraudNet(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_resampled.values, dtype=torch.float32), torch.tensor(y_resampled.values, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
for epoch in range(5):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
torch.save(nn_model.state_dict(), 'nn_model.pth')
print("Models trained and saved successfully.")