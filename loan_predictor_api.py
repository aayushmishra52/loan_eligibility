import pandas as pd
import joblib
import os

class LoanEligibilityPredictor:
    """
    A simple API class for the Loan Eligibility Predictor that can be used in other applications.
    """
    def __init__(self, model_path='loan_eligibility_model.pkl'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        # Check if model exists, if not, train a new one
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Please run loan_eligibility_predictor.py first to train a model.")
            raise FileNotFoundError(f"Model file {model_path} not found")
            
        self.model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    
    def predict(self, applicant_data):
        """
        Predict loan eligibility for a single applicant.
        
        Args:
            applicant_data (dict): Dictionary containing applicant information with the following keys:
                - age: int
                - income: float
                - education: str (one of: 'High School', 'Bachelor', 'Master', 'PhD')
                - employment_years: float
                - credit_score: int
                - loan_amount: float
                - loan_term: int
                - debt_to_income: float
                - has_default_history: int (0 or 1)
                - num_credit_lines: int
        
        Returns:
            dict: Dictionary containing prediction results with the following keys:
                - eligible: bool
                - probability: float
                - recommendation: str
        """
        # Validate input data
        required_fields = [
            'age', 'income', 'education', 'employment_years', 'credit_score',
            'loan_amount', 'loan_term', 'debt_to_income', 'has_default_history', 'num_credit_lines'
        ]
        
        for field in required_fields:
            if field not in applicant_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([applicant_data])
        
        # Get prediction probability
        probability = self.model.predict_proba(input_df)[0, 1]
        eligible = probability >= 0.5
        
        # Generate recommendation based on probability
        if probability >= 0.8:
            recommendation = "Strong candidate for loan approval"
        elif probability >= 0.5:
            recommendation = "Candidate is eligible but with some risk factors"
        elif probability >= 0.3:
            recommendation = "Not eligible, but close to threshold - consider with additional guarantees"
        else:
            recommendation = "Not eligible - significant risk factors present"
        
        return {
            'eligible': eligible,
            'probability': probability,
            'recommendation': recommendation
        }
    
    def batch_predict(self, applicants_data):
        """
        Predict loan eligibility for multiple applicants.
        
        Args:
            applicants_data (list): List of dictionaries, each containing applicant information
        
        Returns:
            list: List of dictionaries, each containing prediction results
        """
        results = []
        for applicant_data in applicants_data:
            try:
                result = self.predict(applicant_data)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results

# Example usage
def example_usage():
    # Create a sample applicant
    sample_applicant = {
        'age': 35,
        'income': 75000,
        'education': 'Bachelor',
        'employment_years': 8,
        'credit_score': 720,
        'loan_amount': 25000,
        'loan_term': 36,
        'debt_to_income': 0.3,
        'has_default_history': 0,
        'num_credit_lines': 3
    }
    
    try:
        # Initialize the predictor
        predictor = LoanEligibilityPredictor()
        
        # Make a prediction
        result = predictor.predict(sample_applicant)
        
        # Print the result
        print("\nPrediction Result:")
        print(f"Eligible: {result['eligible']}")
        print(f"Probability: {result['probability']:.2%}")
        print(f"Recommendation: {result['recommendation']}")
        
        # Batch prediction example
        print("\nBatch Prediction Example:")
        batch_applicants = [
            sample_applicant,  # Good applicant
            {  # Poor applicant
                'age': 22,
                'income': 30000,
                'education': 'High School',
                'employment_years': 1,
                'credit_score': 580,
                'loan_amount': 50000,
                'loan_term': 60,
                'debt_to_income': 0.5,
                'has_default_history': 1,
                'num_credit_lines': 1
            }
        ]
        
        batch_results = predictor.batch_predict(batch_applicants)
        for i, result in enumerate(batch_results):
            print(f"\nApplicant {i+1}:")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Eligible: {result['eligible']}")
                print(f"Probability: {result['probability']:.2%}")
                print(f"Recommendation: {result['recommendation']}")
                
    except FileNotFoundError:
        print("Please run loan_eligibility_predictor.py first to train and save the model.")

if __name__ == "__main__":
    example_usage()