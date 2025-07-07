#!/usr/bin/env python3

import argparse
import os
import sys
import json
from loan_predictor_api import LoanEligibilityPredictor

def parse_arguments():
    """
    Parse command line arguments for the loan eligibility predictor.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Loan Eligibility Predictor CLI')
    
    # Required arguments
    parser.add_argument('--age', type=int, required=True, help='Applicant age (18-100)')
    parser.add_argument('--income', type=float, required=True, help='Annual income in dollars')
    parser.add_argument('--education', type=str, required=True, 
                        choices=['High School', 'Bachelor', 'Master', 'PhD'],
                        help='Education level')
    parser.add_argument('--employment_years', type=float, required=True, 
                        help='Years of employment')
    parser.add_argument('--credit_score', type=int, required=True, 
                        help='Credit score (300-850)')
    parser.add_argument('--loan_amount', type=float, required=True, 
                        help='Requested loan amount in dollars')
    parser.add_argument('--loan_term', type=int, required=True, 
                        choices=[12, 24, 36, 48, 60],
                        help='Loan term in months')
    parser.add_argument('--debt_to_income', type=float, required=True, 
                        help='Debt-to-income ratio (0.1-0.6)')
    parser.add_argument('--has_default_history', type=int, required=True, 
                        choices=[0, 1],
                        help='Has default history (0=No, 1=Yes)')
    parser.add_argument('--num_credit_lines', type=int, required=True, 
                        help='Number of credit lines')
    
    # Optional arguments
    parser.add_argument('--output', type=str, choices=['text', 'json'], default='text',
                        help='Output format (text or json)')
    parser.add_argument('--model', type=str, default='loan_eligibility_model.pkl',
                        help='Path to the model file')
    
    return parser.parse_args()

def validate_arguments(args):
    """
    Validate the command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Returns:
        bool: True if arguments are valid, False otherwise
    """
    # Validate age
    if args.age < 18 or args.age > 100:
        print(f"Error: Age must be between 18 and 100, got {args.age}")
        return False
    
    # Validate income
    if args.income <= 0:
        print(f"Error: Income must be positive, got {args.income}")
        return False
    
    # Validate credit score
    if args.credit_score < 300 or args.credit_score > 850:
        print(f"Error: Credit score must be between 300 and 850, got {args.credit_score}")
        return False
    
    # Validate loan amount
    if args.loan_amount <= 0:
        print(f"Error: Loan amount must be positive, got {args.loan_amount}")
        return False
    
    # Validate debt-to-income ratio
    if args.debt_to_income < 0.1 or args.debt_to_income > 0.6:
        print(f"Error: Debt-to-income ratio must be between 0.1 and 0.6, got {args.debt_to_income}")
        return False
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return False
    
    return True

def format_output(result, output_format):
    """
    Format the prediction result based on the specified output format.
    
    Args:
        result (dict): Prediction result
        output_format (str): Output format ('text' or 'json')
        
    Returns:
        str: Formatted output
    """
    if output_format == 'json':
        # Convert boolean values to integers for JSON serialization
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, bool):
                serializable_result[key] = int(value)
            else:
                serializable_result[key] = value
        return json.dumps(serializable_result, indent=2)
    else:  # text format
        eligible_str = "ELIGIBLE" if result['eligible'] else "NOT ELIGIBLE"
        probability_str = f"{result['probability'] * 100:.2f}%"
        
        output = [
            "=== Loan Eligibility Prediction ===",
            f"Result: {eligible_str}",
            f"Probability: {probability_str}",
            f"Recommendation: {result['recommendation']}",
            "================================="
        ]
        return "\n".join(output)

def main():
    """
    Main function for the CLI predictor.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = LoanEligibilityPredictor(args.model)
        
        # Prepare applicant data
        applicant_data = {
            'age': args.age,
            'income': args.income,
            'education': args.education,
            'employment_years': args.employment_years,
            'credit_score': args.credit_score,
            'loan_amount': args.loan_amount,
            'loan_term': args.loan_term,
            'debt_to_income': args.debt_to_income,
            'has_default_history': args.has_default_history,
            'num_credit_lines': args.num_credit_lines
        }
        
        # Make prediction
        result = predictor.predict(applicant_data)
        
        # Format and print output
        output = format_output(result, args.output)
        print(output)
        
        # Return exit code based on eligibility
        return 0 if result['eligible'] else 1
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())