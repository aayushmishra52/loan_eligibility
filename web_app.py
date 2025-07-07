from flask import Flask, request, jsonify, render_template
import json
from loan_predictor_api import LoanEligibilityPredictor
import os

# Custom JSON encoder to handle boolean values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bool):
            return int(obj)
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder

# Initialize the predictor
try:
    predictor = LoanEligibilityPredictor()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create a modern HTML template for the web interface
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Loan Eligibility Predictor</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
        
        :root {
            --primary-color: #3498db;
            --primary-dark: #2980b9;
            --secondary-color: #2ecc71;
            --secondary-dark: #27ae60;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --text-color: #333;
            --light-text: #777;
            --light-bg: #f9f9f9;
            --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --border-radius: 8px;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
            background-color: #f4f7fa;
            color: var(--text-color);
            min-height: 100vh;
        }
        
        .container {
            width: 90%;
            max-width: 1000px;
            margin: 30px auto;
            padding: 0;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            opacity: 0;
            transform: translateY(-10px);
            transition: all 0.5s ease;
        }
        
        .header h1 {
            color: var(--primary-color);
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .header p {
            color: var(--light-text);
            font-size: 1.1rem;
        }
        
        .card {
            background: white;
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            padding: 30px;
            margin-bottom: 30px;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.5s ease 0.2s;
        }
        
        .card-title {
            color: var(--primary-color);
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .card-title i {
            margin-right: 10px;
        }
        
        .form-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--text-color);
        }
        
        .input-wrapper {
            position: relative;
            transition: all 0.3s ease;
        }
        
        .input-wrapper.focused {
            transform: translateY(-2px);
        }
        
        .error-message {
            color: var(--danger-color);
            font-size: 0.85rem;
            margin-top: 5px;
            padding: 5px 10px;
            border-radius: var(--border-radius);
            background-color: rgba(231, 76, 60, 0.1);
            border-left: 3px solid var(--danger-color);
            animation: fadeIn 0.3s ease-in-out;
            position: absolute;
            z-index: 10;
            right: 0;
            top: 100%;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .icon {
            position: absolute;
            left: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--primary-color);
            transition: all 0.3s ease;
        }
        
        .form-control {
            width: 100%;
            padding: 12px 12px 12px 35px;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
            outline: none;
        }
        
        .form-control.has-value {
            border-color: var(--primary-color);
            background-color: rgba(52, 152, 219, 0.05);
        }
        
        .form-control.invalid {
            border-color: var(--danger-color);
            background-color: rgba(231, 76, 60, 0.05);
        }
        
        .form-control.invalid:focus {
            box-shadow: 0 0 0 3px rgba(231, 76, 60, 0.2);
        }
        
        .form-control.invalid + .icon {
            color: var(--danger-color);
        }
        
        select.form-control {
            appearance: none;
            background-image: url('data:image/svg+xml;utf8,<svg fill="%233498db" height="24" viewBox="0 0 24 24" width="24" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5z"/><path d="M0 0h24v24H0z" fill="none"/></svg>');
            background-repeat: no-repeat;
            background-position: right 10px center;
        }
        
        .btn {
            display: block;
            width: 100%;
            padding: 14px;
            background: linear-gradient(to right, var(--primary-color), var(--primary-dark));
            color: white;
            border: none;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 500;
            margin-top: 30px;
            transition: all 0.3s ease;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .btn:hover {
            background: linear-gradient(to right, var(--primary-dark), var(--primary-color));
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .btn:active {
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        .btn::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 5px;
            height: 5px;
            background: rgba(255, 255, 255, 0.5);
            opacity: 0;
            border-radius: 100%;
            transform: scale(1, 1) translate(-50%);
            transform-origin: 50% 50%;
        }
        
        .btn:focus:not(:active)::after {
            animation: ripple 1s ease-out;
        }
        
        @keyframes ripple {
            0% {
                transform: scale(0, 0);
                opacity: 0.5;
            }
            20% {
                transform: scale(25, 25);
                opacity: 0.3;
            }
            100% {
                opacity: 0;
                transform: scale(40, 40);
            }
        }
        
        .btn i {
            margin-right: 8px;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: var(--border-radius);
            text-align: center;
            box-shadow: var(--card-shadow);
            background-color: white;
            transition: all 0.5s ease;
            transform: translateY(20px);
            opacity: 0;
            animation: fadeIn 0.5s forwards;
        }
        
        @keyframes fadeIn {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .eligible {
            border-top: 5px solid var(--secondary-color);
        }
        
        .eligible h3 {
            color: var(--secondary-color);
        }
        
        .not-eligible {
            border-top: 5px solid var(--danger-color);
        }
        
        .not-eligible h3 {
            color: var(--danger-color);
        }
        
        .hidden {
            display: none;
        }
        
        .progress-container {
            width: 100%;
            background-color: #f1f1f1;
            border-radius: 20px;
            margin-top: 20px;
            overflow: hidden;
            height: 10px;
        }
        
        .progress-bar {
            height: 10px;
            border-radius: 20px;
            background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
            transition: width 1.5s cubic-bezier(0.19, 1, 0.22, 1);
            width: 0;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
            margin-left: 5px;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
            font-weight: normal;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
            color: var(--primary-color);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% {
                opacity: 0.6;
            }
            50% {
                opacity: 1;
            }
            100% {
                opacity: 0.6;
            }
        }
        
        .loading i {
            font-size: 2.5rem;
            margin-bottom: 15px;
            animation: spin 1s infinite linear;
        }
        
        @keyframes spin {
            0% {
                transform: rotate(0deg);
            }
            100% {
                transform: rotate(360deg);
            }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .container {
                width: 95%;
                padding: 10px;
                margin: 15px auto;
            }
            
            .form-container {
                grid-template-columns: 1fr;
            }
            
            .card {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> Loan Eligibility Predictor</h1>
            <p>Enter your details below to check your loan eligibility</p>
        </div>
        
        <div id="model-error" class="result not-eligible hidden">
            <i class="fas fa-exclamation-triangle fa-3x" style="color: var(--danger-color); margin-bottom: 15px;"></i>
            <p>Model not found. Please run loan_eligibility_predictor.py first to train and save the model.</p>
        </div>
        
        <div class="card">
            <h2 class="card-title"><i class="fas fa-user-edit"></i> Applicant Information</h2>
            <form id="loan-form">
                <div class="form-container">
                    <div class="form-group">
                        <label for="age">Age <span class="tooltip"><i class="fas fa-info-circle"></i>
                            <span class="tooltiptext">Applicant's age must be between 18 and 100 years</span>
                        </span></label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-user"></i>
                            <input type="number" id="age" name="age" min="18" max="100" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="income">Annual Income <span class="tooltip"><i class="fas fa-info-circle"></i>
                            <span class="tooltiptext">Your yearly income before taxes</span>
                        </span></label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-dollar-sign"></i>
                            <input type="number" id="income" name="income" min="0" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="education">Education Level</label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-graduation-cap"></i>
                            <select id="education" name="education" class="form-control" required>
                                <option value="High School">High School</option>
                                <option value="Bachelor" selected>Bachelor</option>
                                <option value="Master">Master</option>
                                <option value="PhD">PhD</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="employment_years">Years of Employment</label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-briefcase"></i>
                            <input type="number" id="employment_years" name="employment_years" min="0" step="0.1" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="credit_score">Credit Score <span class="tooltip"><i class="fas fa-info-circle"></i>
                            <span class="tooltiptext">Credit score must be between 300 and 850</span>
                        </span></label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-chart-bar"></i>
                            <input type="number" id="credit_score" name="credit_score" min="300" max="850" class="form-control" required>
                        </div>
                    </div>
                </div>
                
                <h2 class="card-title" style="margin-top: 30px;"><i class="fas fa-file-invoice-dollar"></i> Loan Details</h2>
                <div class="form-container">
                    <div class="form-group">
                        <label for="loan_amount">Loan Amount</label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-money-bill-wave"></i>
                            <input type="number" id="loan_amount" name="loan_amount" min="1000" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="loan_term">Loan Term</label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-calendar-alt"></i>
                            <select id="loan_term" name="loan_term" class="form-control" required>
                                <option value="12">12 months</option>
                                <option value="24">24 months</option>
                                <option value="36" selected>36 months</option>
                                <option value="48">48 months</option>
                                <option value="60">60 months</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="debt_to_income">Debt-to-Income Ratio <span class="tooltip"><i class="fas fa-info-circle"></i>
                            <span class="tooltiptext">The percentage of your monthly income that goes toward paying debts (0.1-0.6)</span>
                        </span></label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-percentage"></i>
                            <input type="number" id="debt_to_income" name="debt_to_income" min="0.1" max="0.6" step="0.01" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="has_default_history">Default History</label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-history"></i>
                            <select id="has_default_history" name="has_default_history" class="form-control" required>
                                <option value="0" selected>No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="num_credit_lines">Number of Credit Lines</label>
                        <div class="input-wrapper">
                            <i class="icon fas fa-credit-card"></i>
                            <input type="number" id="num_credit_lines" name="num_credit_lines" min="0" class="form-control" required>
                        </div>
                    </div>
                </div>
                
                <button type="submit" class="btn"><i class="fas fa-calculator"></i> Predict Eligibility</button>
            </form>
        </div>
        
        <div id="loading" class="loading">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Analyzing your application...</p>
        </div>
        
        <div id="result" class="result hidden">
            <h3 id="result-title"></h3>
            <p id="result-probability"></p>
            <p id="result-recommendation"></p>
            <div class="progress-container">
                <div id="progress-bar" class="progress-bar"></div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const loanForm = document.getElementById('loan-form');
            const resultDiv = document.getElementById('result');
            const resultTitle = document.getElementById('result-title');
            const resultProb = document.getElementById('result-probability');
            const resultRecommendation = document.getElementById('result-recommendation');
            const progressBar = document.getElementById('progress-bar');
            const loadingDiv = document.getElementById('loading');
            const modelError = document.getElementById('model-error');
            
            // Animate elements on page load
            setTimeout(() => {
                document.querySelector('.header').style.opacity = '1';
                document.querySelector('.card').style.opacity = '1';
            }, 100);
            
            // Check if model is available
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (!data.model_loaded) {
                        modelError.classList.remove('hidden');
                        loanForm.classList.add('hidden');
                    }
                })
                .catch(error => {
                    console.error('Error checking model status:', error);
                    modelError.classList.remove('hidden');
                    modelError.innerHTML = '<i class="fas fa-exclamation-triangle fa-3x" style="color: var(--danger-color); margin-bottom: 15px;"></i><p>Error connecting to server. Please try again later.</p>';
                });
            
            // Add input validation and formatting
            const inputs = document.querySelectorAll('input[type="number"]');
            inputs.forEach(input => {
                input.addEventListener('input', function() {
                    if (this.value !== '') {
                        this.classList.add('has-value');
                        
                        // Validate input ranges
                        const id = this.id;
                        const value = parseFloat(this.value);
                        let isValid = true;
                        
                        // Specific validation rules
                        if (id === 'age' && (value < 18 || value > 100)) {
                            isValid = false;
                        } else if (id === 'income' && value < 0) {
                            isValid = false;
                        } else if (id === 'employment_years' && value < 0) {
                            isValid = false;
                        } else if (id === 'credit_score' && (value < 300 || value > 850)) {
                            isValid = false;
                        } else if (id === 'loan_amount' && value < 1000) {
                            isValid = false;
                        } else if (id === 'debt_to_income' && (value < 0.1 || value > 0.6)) {
                            isValid = false;
                        } else if (id === 'num_credit_lines' && (value < 0 || !Number.isInteger(value))) {
                            isValid = false;
                        }
                        
                        if (!isValid) {
                            this.classList.add('invalid');
                            this.setCustomValidity('Please enter a valid value');
                        } else {
                            this.classList.remove('invalid');
                            this.setCustomValidity('');
                        }
                    } else {
                        this.classList.remove('has-value');
                        this.classList.remove('invalid');
                        this.setCustomValidity('');
                    }
                });
            });
            
            // Validate all form inputs and show error messages
            function validateForm() {
                const inputs = document.querySelectorAll('.form-control');
                let isValid = true;
                let firstInvalidInput = null;
                
                inputs.forEach(input => {
                    // Trigger the input event to validate
                    const event = new Event('input', { bubbles: true });
                    input.dispatchEvent(event);
                    
                    // Check if input is valid
                    if (input.validity.valid === false) {
                        isValid = false;
                        if (!firstInvalidInput) {
                            firstInvalidInput = input;
                        }
                    }
                });
                
                // Focus on the first invalid input
                if (firstInvalidInput) {
                    firstInvalidInput.focus();
                    // Show a tooltip or message near the invalid field
                    const fieldName = firstInvalidInput.previousElementSibling ? 
                                     firstInvalidInput.previousElementSibling.textContent : 
                                     firstInvalidInput.name;
                    
                    // Create or update error message
                    let errorMsg = document.getElementById('form-error-message');
                    if (!errorMsg) {
                        errorMsg = document.createElement('div');
                        errorMsg.id = 'form-error-message';
                        errorMsg.className = 'error-message';
                        firstInvalidInput.parentElement.appendChild(errorMsg);
                    }
                    errorMsg.textContent = `Please enter a valid ${fieldName.trim().replace('*', '')}`;
                    errorMsg.style.display = 'block';
                    
                    // Hide error message after 3 seconds
                    setTimeout(() => {
                        if (errorMsg) {
                            errorMsg.style.display = 'none';
                        }
                    }, 3000);
                }
                
                return isValid;
            }
            
            // Handle form submission
            loanForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Validate form before submission
                if (!validateForm()) {
                    return false;
                }
                
                // Show loading indicator
                loadingDiv.style.display = 'block';
                resultDiv.classList.add('hidden');
                
                // Hide any error messages
                const errorMsgs = document.querySelectorAll('.error-message');
                errorMsgs.forEach(msg => {
                    msg.style.display = 'none';
                });
                
                // Get form data
                const formData = {
                    age: parseInt(document.getElementById('age').value),
                    income: parseFloat(document.getElementById('income').value),
                    education: document.getElementById('education').value,
                    employment_years: parseFloat(document.getElementById('employment_years').value),
                    credit_score: parseInt(document.getElementById('credit_score').value),
                    loan_amount: parseFloat(document.getElementById('loan_amount').value),
                    loan_term: parseInt(document.getElementById('loan_term').value),
                    debt_to_income: parseFloat(document.getElementById('debt_to_income').value),
                    has_default_history: parseInt(document.getElementById('has_default_history').value),
                    num_credit_lines: parseInt(document.getElementById('num_credit_lines').value)
                };
                
                fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                })
                .then(response => {
                    // Always parse the JSON first, even if response is not OK
                    return response.json().then(data => {
                        // If response is not OK, throw an error with the error message from the server
                        if (!response.ok) {
                            if (data && data.error) {
                                throw new Error(data.error);
                            } else {
                                throw new Error('Network response was not ok');
                            }
                        }
                        return data;
                    });
                })
                .then(data => {
                    // Hide loading indicator
                    loadingDiv.style.display = 'none';
                    
                    resultDiv.classList.remove('hidden');
                    
                    // Set initial width to 0 for animation
                    progressBar.style.width = '0%';
                    
                    // Check if eligible (could be boolean or integer 1/0)
                    const isEligible = data.eligible === true || data.eligible === 1;
                    
                    if (isEligible) {
                        resultDiv.className = 'result eligible';
                        resultTitle.innerHTML = '<i class="fas fa-check-circle"></i> Eligible for Loan';
                        progressBar.style.backgroundColor = '#28a745';
                    } else {
                        resultDiv.className = 'result not-eligible';
                        resultTitle.innerHTML = '<i class="fas fa-times-circle"></i> Not Eligible for Loan';
                        progressBar.style.backgroundColor = '#dc3545';
                    }
                    
                    // Set probability and recommendation
                    resultProb.innerHTML = `<strong>Approval Probability:</strong> ${(data.probability * 100).toFixed(2)}%`;
                    resultRecommendation.innerHTML = `<strong>Recommendation:</strong> ${data.recommendation}`;
                    
                    // Animate the progress bar
                    setTimeout(() => {
                        progressBar.style.width = `${data.probability * 100}%`;
                        progressBar.textContent = `${(data.probability * 100).toFixed(0)}%`;
                    }, 100);
                    
                    // Scroll to result
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                })
                .catch(error => {
                    console.error('Error:', error);
                    loadingDiv.style.display = 'none';
                    
                    // Show error message
                    resultDiv.classList.remove('hidden');
                    resultDiv.className = 'result not-eligible';
                    resultTitle.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
                    resultProb.textContent = '';
                    
                    // Display the specific error message if available
                    const errorMessage = error.message || 'An error occurred while processing your request. Please try again.';
                    resultRecommendation.textContent = errorMessage;
                    
                    // Scroll to result
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                });
            });
            
            // Add some visual feedback on form fields
            const formControls = document.querySelectorAll('.form-control');
            formControls.forEach(control => {
                control.addEventListener('focus', function() {
                    this.parentElement.classList.add('focused');
                });
                
                control.addEventListener('blur', function() {
                    this.parentElement.classList.remove('focused');
                });
            });
        });
    </script>
</body>
</html>
''')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({'model_loaded': model_loaded})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get applicant data from request
    applicant_data = request.json
    
    try:
        # Validate input data before prediction
        required_fields = [
            'age', 'income', 'education', 'employment_years', 'credit_score',
            'loan_amount', 'loan_term', 'debt_to_income', 'has_default_history', 'num_credit_lines'
        ]
        
        for field in required_fields:
            if field not in applicant_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            
            # Ensure numeric fields are valid numbers
            if field != 'education':
                try:
                    if field in ['has_default_history']:
                        applicant_data[field] = int(applicant_data[field])
                    elif field in ['age', 'credit_score', 'loan_term', 'num_credit_lines']:
                        applicant_data[field] = int(applicant_data[field])
                    else:
                        applicant_data[field] = float(applicant_data[field])
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid value for {field}'}), 400
        
        # Make prediction
        result = predictor.predict(applicant_data)
        
        # Create a new dictionary with JSON-serializable values
        serializable_result = {
            'eligible': 1 if result['eligible'] else 0,
            'probability': result['probability'],
            'recommendation': result['recommendation']
        }
            
        return jsonify(serializable_result)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Add Flask to requirements.txt if it doesn't exist
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    if 'flask' not in requirements.lower():
        with open('requirements.txt', 'a') as f:
            f.write('\nflask>=2.0.0')
    
    print("Starting Flask web server...")
    print("Open http://127.0.0.1:5000 in your browser to use the web application")
    app.run(debug=True)