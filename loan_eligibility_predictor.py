import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Function to generate synthetic loan application data
def generate_synthetic_data(n_samples=1000):
    data = {
        'age': np.random.randint(18, 75, n_samples),
        'income': np.random.randint(20000, 200000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.randint(5000, 100000, n_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.6, n_samples),
        'has_default_history': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'num_credit_lines': np.random.randint(0, 15, n_samples)
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to make it more realistic
    for col in ['income', 'employment_years', 'credit_score']:
        mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan
    
    # Generate the target variable (loan_approved) based on some rules
    # Higher probability of approval for higher income, credit score, and education level
    prob_approval = (
        0.4 + 
        0.2 * (df['income'] > 50000) + 
        0.2 * (df['credit_score'] > 650) + 
        0.1 * (df['education'].isin(['Master', 'PhD'])) + 
        0.1 * (df['employment_years'] > 5) - 
        0.3 * df['has_default_history'] - 
        0.2 * (df['debt_to_income'] > 0.4)
    )
    
    # Ensure probabilities are between 0 and 1
    prob_approval = np.clip(prob_approval, 0, 1)
    
    # Generate loan_approved based on calculated probabilities
    df['loan_approved'] = np.random.binomial(1, prob_approval)
    
    return df

# Generate and save the dataset
def create_dataset():
    print("Generating synthetic loan application dataset...")
    df = generate_synthetic_data(1000)
    df.to_csv('loan_application_data.csv', index=False)
    print(f"Dataset created with {len(df)} samples and saved to 'loan_application_data.csv'")
    return df

# Data preprocessing function
def preprocess_data(df):
    print("Preprocessing data...")
    
    # Split features and target
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Define preprocessing for numerical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, preprocessor

# Train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    print("Training and evaluating models...")
    
    # Create and train Logistic Regression model
    log_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', LogisticRegression(max_iter=1000))])
    log_reg_pipeline.fit(X_train, y_train)
    
    # Create and train Random Forest model
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(n_estimators=100))])
    rf_pipeline.fit(X_train, y_train)
    
    # Evaluate models
    models = {
        'Logistic Regression': log_reg_pipeline,
        'Random Forest': rf_pipeline
    }
    
    results = {}
    plt.figure(figsize=(12, 10))
    
    for i, (name, model) in enumerate(models.items()):
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.subplot(2, 2, i+1)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(2, 2, i+3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'roc_auc': roc_auc
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Precision: {results[name]['precision']:.4f}")
        print(f"Recall: {results[name]['recall']:.4f}")
        print(f"F1 Score: {results[name]['f1']:.4f}")
        print(f"ROC AUC: {results[name]['roc_auc']:.4f}")
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    print("Evaluation plots saved to 'model_evaluation.png'")
    
    # Save the best model based on ROC AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, 'loan_eligibility_model.pkl')
    print(f"Best model ({best_model_name}) saved to 'loan_eligibility_model.pkl'")
    
    return best_model, results

# Create a simple GUI for loan eligibility prediction
class LoanEligibilityApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Loan Eligibility Predictor")
        self.root.geometry("600x650")
        self.root.resizable(False, False)
        
        # Set style
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 12))
        style.configure('TButton', font=('Arial', 12))
        style.configure('TEntry', font=('Arial', 12))
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Loan Eligibility Predictor", font=('Arial', 18, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Input fields
        self.create_input_field(main_frame, "Age:", 1, "age")
        self.create_input_field(main_frame, "Annual Income ($):", 2, "income")
        
        # Education dropdown
        ttk.Label(main_frame, text="Education Level:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.education_var = tk.StringVar()
        education_combo = ttk.Combobox(main_frame, textvariable=self.education_var, width=20)
        education_combo['values'] = ('High School', 'Bachelor', 'Master', 'PhD')
        education_combo.grid(row=3, column=1, sticky=tk.W, pady=5)
        education_combo.current(1)  # Default to Bachelor
        
        self.create_input_field(main_frame, "Years of Employment:", 4, "employment_years")
        self.create_input_field(main_frame, "Credit Score (300-850):", 5, "credit_score")
        self.create_input_field(main_frame, "Loan Amount ($):", 6, "loan_amount")
        
        # Loan term dropdown
        ttk.Label(main_frame, text="Loan Term (months):").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.loan_term_var = tk.StringVar()
        loan_term_combo = ttk.Combobox(main_frame, textvariable=self.loan_term_var, width=20)
        loan_term_combo['values'] = ('12', '24', '36', '48', '60')
        loan_term_combo.grid(row=7, column=1, sticky=tk.W, pady=5)
        loan_term_combo.current(2)  # Default to 36 months
        
        self.create_input_field(main_frame, "Debt-to-Income Ratio (0.1-0.6):", 8, "debt_to_income")
        
        # Default history
        ttk.Label(main_frame, text="Has Default History:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.default_history_var = tk.IntVar()
        default_check = ttk.Checkbutton(main_frame, variable=self.default_history_var)
        default_check.grid(row=9, column=1, sticky=tk.W, pady=5)
        
        self.create_input_field(main_frame, "Number of Credit Lines:", 10, "num_credit_lines")
        
        # Predict button
        predict_btn = ttk.Button(main_frame, text="Predict Eligibility", command=self.predict_eligibility)
        predict_btn.grid(row=11, column=0, columnspan=2, pady=20)
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding="10")
        result_frame.grid(row=12, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        self.result_label = ttk.Label(result_frame, text="Fill the form and click 'Predict Eligibility'", font=('Arial', 12))
        self.result_label.pack(pady=10)
        
        self.probability_label = ttk.Label(result_frame, text="")
        self.probability_label.pack(pady=5)
        
        # Progress bar for visualization
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(result_frame, orient=tk.HORIZONTAL, length=400, mode='determinate', variable=self.progress_var)
        self.progress.pack(pady=10)
        
        # Input validation
        self.root.bind('<Return>', lambda event: self.predict_eligibility())
    
    def create_input_field(self, parent, label_text, row, var_name):
        ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=5)
        var = tk.StringVar()
        setattr(self, f"{var_name}_var", var)
        ttk.Entry(parent, textvariable=var, width=20).grid(row=row, column=1, sticky=tk.W, pady=5)
    
    def predict_eligibility(self):
        try:
            # Get input values
            input_data = {
                'age': int(self.age_var.get()),
                'income': float(self.income_var.get()),
                'education': self.education_var.get(),
                'employment_years': float(self.employment_years_var.get()),
                'credit_score': int(self.credit_score_var.get()),
                'loan_amount': float(self.loan_amount_var.get()),
                'loan_term': int(self.loan_term_var.get()),
                'debt_to_income': float(self.debt_to_income_var.get()),
                'has_default_history': self.default_history_var.get(),
                'num_credit_lines': int(self.num_credit_lines_var.get())
            }
            
            # Validate inputs
            if input_data['age'] < 18 or input_data['age'] > 100:
                raise ValueError("Age must be between 18 and 100")
            if input_data['credit_score'] < 300 or input_data['credit_score'] > 850:
                raise ValueError("Credit score must be between 300 and 850")
            if input_data['debt_to_income'] < 0.1 or input_data['debt_to_income'] > 0.6:
                raise ValueError("Debt-to-income ratio must be between 0.1 and 0.6")
            
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Get prediction probability
            probability = self.model.predict_proba(input_df)[0, 1]
            
            # Update progress bar
            self.progress_var.set(probability * 100)
            
            # Update result labels
            if probability >= 0.5:
                self.result_label.config(text="Eligible for Loan", foreground="green")
            else:
                self.result_label.config(text="Not Eligible for Loan", foreground="red")
            
            self.probability_label.config(text=f"Approval Probability: {probability:.2%}")
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Main function
def main(headless=False):
    # Check if dataset exists, if not create it
    try:
        df = pd.read_csv('loan_application_data.csv')
        print(f"Loaded existing dataset with {len(df)} samples")
    except FileNotFoundError:
        df = create_dataset()
    
    # Display dataset info
    print("\nDataset Information:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    print("\nTarget distribution:")
    print(df['loan_approved'].value_counts(normalize=True))
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train and evaluate models
    best_model, results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Launch GUI only if not in headless mode
    if not headless:
        print("\nLaunching Loan Eligibility Predictor GUI...")
        root = tk.Tk()
        app = LoanEligibilityApp(root, best_model)
        root.mainloop()
    else:
        print("\nRunning in headless mode. Model saved successfully.")

if __name__ == "__main__":
    # Run in headless mode to avoid GUI issues
    main(headless=True)