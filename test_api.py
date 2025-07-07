from web_app import app, predictor
import json

# Create a test client using Flask's test_client
client = app.test_client()

# Define the loan application data
applicant_data = {
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

# Test direct prediction from the predictor
print("\nDirect prediction from predictor:")
result_direct = predictor.predict(applicant_data)
print(f"Type of 'eligible' from direct prediction: {type(result_direct['eligible']).__name__}")
print(f"Value of 'eligible' from direct prediction: {result_direct['eligible']}")

# Make the POST request using Flask's test_client
print("\nAPI test using Flask test_client:")
response = client.post('/api/predict', json=applicant_data)

# Print the status code
print(f"Status Code: {response.status_code}")

# Try to parse the response as JSON
try:
    result = response.get_json()
    print("Response JSON:")
    print(json.dumps(result, indent=4))
    
    # Check the type of 'eligible' field
    if 'eligible' in result:
        print(f"Type of 'eligible' from API: {type(result['eligible']).__name__}")
        print(f"Value of 'eligible' from API: {result['eligible']}")
    else:
        print("No 'eligible' field found in response")
except Exception as e:
    print(f"Error parsing JSON: {str(e)}")
    print("Response Content:")
    print(response.data.decode('utf-8'))