# Loan Eligibility Predictor

## Overview
This project implements a machine learning-based loan eligibility prediction system. It uses classification models to predict whether a loan application should be approved based on various applicant details such as age, income, education level, credit score, and more.

## Features
- **Synthetic Data Generation**: Creates realistic loan application data for training and testing
- **Data Preprocessing**: Handles missing values and encodes categorical features
- **Model Training**: Implements both Logistic Regression and Random Forest classifiers
- **Model Evaluation**: Evaluates models using ROC curves, confusion matrices, and classification reports
- **User-friendly GUI**: Provides an intuitive interface for making predictions on new applicant data
- **API Integration**: Offers a simple API for integration with other applications
- **Web Application**: Includes a Flask-based web interface for online predictions

## Requirements
- Python 3.6+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, tkinter, flask

## Installation
1. Clone this repository or download the files
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   Note: tkinter usually comes pre-installed with Python

## Usage

### Desktop Application
1. Run the main script:
   ```
   python loan_eligibility_predictor.py
   ```
2. The program will:
   - Generate synthetic data if no dataset exists
   - Preprocess the data
   - Train and evaluate Logistic Regression and Random Forest models
   - Save the best model and evaluation plots
   - Launch the GUI for making predictions

3. In the GUI, enter applicant details and click "Predict Eligibility" to see the result

### API Usage
1. First, run the main script to generate the model:
   ```
   python loan_eligibility_predictor.py
   ```

2. Then, you can use the API in your applications:
   ```
   python loan_predictor_api.py
   ```
   This will demonstrate how to use the API with sample applicants.

### Web Application
1. First, run the main script to generate the model:
   ```
   python loan_eligibility_predictor.py
   ```

2. Then, start the web application:
   ```
   python web_app.py
   ```

3. Open your browser and navigate to http://127.0.0.1:5000 to use the web interface

### Batch Processing
1. First, run the main script to generate the model:
   ```
   python loan_eligibility_predictor.py
   ```

2. Then, run the batch processor:
   ```
   python batch_processor.py
   ```

3. The script will:
   - Generate synthetic loan applications (or use a provided CSV file)
   - Process all applications using the trained model
   - Save results to 'batch_results.csv'
   - Generate summary visualizations in 'batch_processing_report.png'

### Command Line Interface
1. First, run the main script to generate the model:
   ```
   python loan_eligibility_predictor.py
   ```

2. Then, use the CLI for quick predictions:
   ```
   python cli_predictor.py --age 35 --income 75000 --education "Bachelor" --employment_years 8 --credit_score 720 --loan_amount 25000 --loan_term 36 --debt_to_income 0.3 --has_default_history 0 --num_credit_lines 3
   ```

3. For JSON output, add the `--output json` flag:
   ```
   python cli_predictor.py --age 35 --income 75000 --education "Bachelor" --employment_years 8 --credit_score 720 --loan_amount 25000 --loan_term 36 --debt_to_income 0.3 --has_default_history 0 --num_credit_lines 3 --output json
   ```

## Files
- `loan_eligibility_predictor.py`: Main script containing all functionality
- `loan_predictor_api.py`: API for integrating the predictor with other applications
- `web_app.py`: Flask-based web application for online predictions
- `batch_processor.py`: Script for batch processing of multiple loan applications
- `cli_predictor.py`: Command-line interface for quick predictions
- `requirements.txt`: List of required Python packages
- `loan_application_data.csv`: Generated synthetic dataset (created on first run)
- `loan_eligibility_model.pkl`: Saved best-performing model (created after training)
- `model_evaluation.png`: Visualization of model performance (created after evaluation)
- `batch_results.csv`: Results from batch processing (created when running batch_processor.py)
- `batch_processing_report.png`: Visualizations of batch processing results

## How It Works
1. **Data Generation**: Creates synthetic data with realistic relationships between features and loan approval
2. **Preprocessing**: Handles missing values, scales numerical features, and encodes categorical variables
3. **Model Training**: Trains Logistic Regression and Random Forest models
4. **Evaluation**: Compares models using various metrics and selects the best performer
5. **Prediction**: Uses the trained model to predict loan eligibility for new applicants

## Project Structure
- Data generation and preprocessing
- Model training and evaluation
- GUI implementation for user interaction
- API for integration with other applications
- Web application for online predictions
- Batch processing for multiple applications
- Command-line interface for quick predictions

## Future Improvements
- Add more classification algorithms (SVM, Neural Networks, etc.)
- Implement feature importance analysis
- Add data visualization in the GUI
- Allow users to save and load prediction results
- Implement cross-validation for more robust model evaluation