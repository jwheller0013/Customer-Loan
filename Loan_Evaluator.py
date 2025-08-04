import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np


file_path = 'Task 3 and 4_Loan_Data.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")

features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']
target = 'default'

X = df[features]
y = df[target]

#Create training and testing sets#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Create models and store model performance#
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
}

model_performance = {}
best_model = None
best_auc_roc = -1
best_model_name = None

#Testing models#
print("\nTraining and evaluating models...")
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    try:
        model.fit(X_train, y_train)
        
        # Predict probabilities on the test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of default (class 1)
        
        # Evaluate using AUC-ROC
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        model_performance[name] = {'AUC-ROC': auc_roc}
        print(f"{name} AUC-ROC: {auc_roc:.4f}")
        
        # Classification report for detailed metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        model_performance[name]['Classification Report'] = report
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Track best model
        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_model = model
            best_model_name = name
            
    except Exception as e:
        print(f"Error training {name}: {e}")

if best_model is None:
    print("Error: No models were successfully trained.")
    exit()

print(f"\nBest performing model based on AUC-ROC: {best_model_name} (AUC-ROC: {best_auc_roc:.4f})")

# Define the recovery rate and Loss Given Default (LGD)
RECOVERY_RATE = 0.10
LGD = 1 - RECOVERY_RATE

