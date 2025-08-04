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

def calculate_expected_loss(loan_properties: dict, model, lgd: float, features_order: list) -> dict:
    """
    Calculates the Expected Loss (EL) for a given loan.

    Args:
        loan_properties (dict): A dictionary containing the loan's features.
        model: The trained machine learning model used to predict the Probability of Default (PD).
        lgd (float): Loss Given Default (1 - Recovery Rate).
        features_order (list): A list of feature names in the order expected by the model.

    Returns:
        dict: A dictionary containing EL, PD, EAD, and other relevant metrics.
    """
    try:
        # Validate required fields
        required_fields = set(features_order)
        provided_fields = set(loan_properties.keys())
        missing_fields = required_fields - provided_fields
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Convert loan_properties to a DataFrame row for prediction
        loan_df = pd.DataFrame([loan_properties])
        
        # Ensure the order of columns matches the training data features
        loan_df = loan_df[features_order]
        
        # Predict the Probability of Default (PD)
        pd_value = model.predict_proba(loan_df)[0, 1]  # Probability of default (class 1)
        
        # Get the Exposure at Default (EAD)
        ead = loan_properties['loan_amt_outstanding']
        
        # Calculate Expected Loss (EL)
        el = pd_value * lgd * ead
        
        return {
            'expected_loss': el,
            'probability_of_default': pd_value,
            'loss_given_default': lgd,
            'exposure_at_default': ead,
            'recovery_rate': 1 - lgd
        }
        
    except Exception as e:
        raise ValueError(f"Error calculating expected loss: {e}")
    
# Example usage with sample loan data
# sample_loan_data = {
#     'credit_lines_outstanding': 2,
#     'loan_amt_outstanding': 15000.0,
#     'total_debt_outstanding': 5000.0,
#     'income': 75000.0,
#     'years_employed': 5,
#     'fico_score': 680
# }

# print("\n--- Example Expected Loss Calculation ---")
# try:
#     el_results = calculate_expected_loss(sample_loan_data, best_model, LGD, features)
#     print(f"Sample Loan Data: {sample_loan_data}")
#     print(f"\nResults:")
#     print(f"Probability of Default (PD): {el_results['probability_of_default']:.4f} ({el_results['probability_of_default']:.2%})")
#     print(f"Loss Given Default (LGD): {el_results['loss_given_default']:.4f}")
#     print(f"Exposure at Default (EAD): ${el_results['exposure_at_default']:,.2f}")
#     print(f"Expected Loss (EL): ${el_results['expected_loss']:,.2f}")
    
# except Exception as e:
#     print(f"Error in example calculation: {e}")

def get_user_loan_input():
    """
    Interactive function to get loan details from user input.
    
    Returns:
        dict: Dictionary containing loan properties
    """
    print("\n" + "="*60)
    print("LOAN DEFAULT RISK ASSESSMENT")
    print("="*60)
    print("Please enter the following loan details:")
    
    # Feature descriptions and validation ranges
    feature_info = {
        'credit_lines_outstanding': {
            'description': 'Number of credit lines outstanding',
            'type': int,
            'min_val': 0,
            'max_val': 50
        },
        'loan_amt_outstanding': {
            'description': 'Loan amount outstanding ($)',
            'type': float,
            'min_val': 0,
            'max_val': 1000000
        },
        'total_debt_outstanding': {
            'description': 'Total debt outstanding ($)',
            'type': float,
            'min_val': 0,
            'max_val': 2000000
        },
        'income': {
            'description': 'Annual income ($)',
            'type': float,
            'min_val': 0,
            'max_val': 10000000
        },
        'years_employed': {
            'description': 'Years of employment',
            'type': int,
            'min_val': 0,
            'max_val': 60
        },
        'fico_score': {
            'description': 'FICO credit score',
            'type': int,
            'min_val': 300,
            'max_val': 850
        }
    }
    
    loan_data = {}
    
    for feature in features:
        while True:
            try:
                info = feature_info[feature]
                prompt = f"\n{info['description']} ({info['min_val']}-{info['max_val']}): "
                
                if info['type'] == int:
                    value = int(input(prompt))
                else:
                    value = float(input(prompt))
                
                # Validate range
                if info['min_val'] <= value <= info['max_val']:
                    loan_data[feature] = value
                    break
                else:
                    print(f"Please enter a value between {info['min_val']} and {info['max_val']}")
                    
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return None
    
    return loan_data

def assess_loan_risk(loan_data):
    """
    Assess the risk of a single loan and display results.
    
    Args:
        loan_data (dict): Loan properties
    
    Returns:
        dict: Assessment results
    """
    try:
        results = calculate_expected_loss(loan_data, best_model, LGD, features)
        
        print(f"\n" + "-"*50)
        print("LOAN RISK ASSESSMENT RESULTS")
        print("-"*50)
        
        # Risk categorization
        pd_value = results['probability_of_default']
        if pd_value < 0.05:
            risk_level = "LOW"
        elif pd_value < 0.15:
            risk_level = "MODERATE"
        elif pd_value < 0.30:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        print(f"Risk Level: {risk_level}")
        print(f"Probability of Default: {pd_value:.2%}")
        print(f"Expected Loss: ${results['expected_loss']:,.2f}")
        print(f"Loan Amount: ${results['exposure_at_default']:,.2f}")
        print(f"Loss Rate: {results['expected_loss']/results['exposure_at_default']:.2%}")
        
        # Recommendation
        if pd_value < 0.10:
            recommendation = "APPROVE - Low risk loan"
        elif pd_value < 0.20:
            recommendation = "CONDITIONAL APPROVAL - Consider higher interest rate"
        else:
            recommendation = "DECLINE - High risk of default"
        
        print(f"\nRecommendation: {recommendation}")
        
        return results
        
    except Exception as e:
        print(f"Error assessing loan: {e}")
        return None

loan_data = get_user_loan_input()
assess_loan_risk(loan_data)