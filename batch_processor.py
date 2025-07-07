import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from loan_predictor_api import LoanEligibilityPredictor

def generate_batch_data(n_samples=100):
    """
    Generate a batch of synthetic loan applications for demonstration purposes.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: DataFrame containing the generated data
    """
    print(f"Generating {n_samples} synthetic loan applications...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random data
    data = []
    for _ in range(n_samples):
        applicant = {
            'applicant_id': f"APP-{np.random.randint(10000, 99999)}",
            'age': np.random.randint(18, 75),
            'income': np.random.randint(20000, 200000),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
            'employment_years': np.random.randint(0, 40),
            'credit_score': np.random.randint(300, 850),
            'loan_amount': np.random.randint(5000, 100000),
            'loan_term': np.random.choice([12, 24, 36, 48, 60]),
            'debt_to_income': np.random.uniform(0.1, 0.6),
            'has_default_history': np.random.choice([0, 1], p=[0.9, 0.1]),
            'num_credit_lines': np.random.randint(0, 15)
        }
        data.append(applicant)
    
    return pd.DataFrame(data)

def process_batch(input_file=None, output_file='batch_results.csv'):
    """
    Process a batch of loan applications and save the results.
    
    Args:
        input_file (str, optional): Path to CSV file containing loan applications.
                                   If None, generates synthetic data.
        output_file (str): Path to save the results
        
    Returns:
        pd.DataFrame: DataFrame containing the original data and prediction results
    """
    # Check if model exists
    if not os.path.exists('loan_eligibility_model.pkl'):
        print("Model not found. Please run loan_eligibility_predictor.py first to train a model.")
        return None
    
    # Load or generate data
    if input_file and os.path.exists(input_file):
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
    else:
        df = generate_batch_data()
    
    print(f"Processing {len(df)} loan applications...")
    
    # Initialize predictor
    predictor = LoanEligibilityPredictor()
    
    # Process each application
    results = []
    for _, row in df.iterrows():
        # Extract required fields for prediction
        applicant_data = {
            'age': row['age'],
            'income': row['income'],
            'education': row['education'],
            'employment_years': row['employment_years'],
            'credit_score': row['credit_score'],
            'loan_amount': row['loan_amount'],
            'loan_term': row['loan_term'],
            'debt_to_income': row['debt_to_income'],
            'has_default_history': row['has_default_history'],
            'num_credit_lines': row['num_credit_lines']
        }
        
        # Make prediction
        prediction = predictor.predict(applicant_data)
        
        # Combine original data with prediction results
        result = row.to_dict()
        
        # Convert boolean values to integers for serialization
        eligible_value = 1 if prediction['eligible'] else 0
        
        result.update({
            'eligible': eligible_value,
            'probability': prediction['probability'],
            'recommendation': prediction['recommendation']
        })
        
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Generate summary statistics
    generate_summary_report(results_df)
    
    return results_df

def generate_summary_report(df):
    """
    Generate summary statistics and visualizations for batch processing results.
    
    Args:
        df (pd.DataFrame): DataFrame containing the batch processing results
    """
    print("\nGenerating summary report...")
    
    # Calculate approval rate
    approval_rate = df['eligible'].mean() * 100
    print(f"Overall approval rate: {approval_rate:.2f}%")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Approval distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x='eligible', data=df)
    plt.title('Loan Approval Distribution')
    plt.xlabel('Approved')
    plt.xticks([0, 1], ['No', 'Yes'])
    
    # 2. Approval probability distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['probability'], bins=20, kde=True)
    plt.title('Approval Probability Distribution')
    plt.xlabel('Probability')
    
    # 3. Approval rate by education
    plt.subplot(2, 2, 3)
    education_approval = df.groupby('education')['eligible'].mean().sort_values() * 100
    sns.barplot(x=education_approval.index, y=education_approval.values)
    plt.title('Approval Rate by Education')
    plt.xlabel('Education')
    plt.ylabel('Approval Rate (%)')
    plt.xticks(rotation=45)
    
    # 4. Credit score vs. Approval probability
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='credit_score', y='probability', hue='eligible', data=df)
    plt.title('Credit Score vs. Approval Probability')
    plt.xlabel('Credit Score')
    plt.ylabel('Approval Probability')
    
    plt.tight_layout()
    plt.savefig('batch_processing_report.png')
    print("Summary visualizations saved to 'batch_processing_report.png'")
    
    # Additional statistics
    print("\nApproval rate by education:")
    print(education_approval)
    
    print("\nApproval rate by default history:")
    default_approval = df.groupby('has_default_history')['eligible'].mean() * 100
    print(f"No default history: {default_approval[0]:.2f}%")
    print(f"Has default history: {default_approval[1]:.2f}%")
    
    # Calculate average values for approved vs. rejected applications
    print("\nAverage values for approved vs. rejected applications:")
    numeric_cols = ['age', 'income', 'employment_years', 'credit_score', 
                   'loan_amount', 'debt_to_income', 'num_credit_lines']
    
    avg_by_approval = df.groupby('eligible')[numeric_cols].mean()
    print(avg_by_approval)

def main():
    """
    Main function to demonstrate batch processing.
    """
    print("Loan Eligibility Predictor - Batch Processing")
    print("===========================================\n")
    
    # Check if model exists
    if not os.path.exists('loan_eligibility_model.pkl'):
        print("Model not found. Please run loan_eligibility_predictor.py first to train a model.")
        return
    
    # Process batch
    process_batch()
    
    print("\nBatch processing complete!")
    print("You can now analyze the results in 'batch_results.csv' and 'batch_processing_report.png'")

if __name__ == "__main__":
    main()