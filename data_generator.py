import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_accounting_data(n_samples=1000, seed=42):
    """Generate synthetic accounting dataset for ML demonstrations"""
    np.random.seed(seed)
    
    # Client demographics
    client_ages = np.random.randint(22, 70, n_samples)
    annual_incomes = np.random.normal(75000, 30000, n_samples).clip(20000, 250000)
    
    # Transaction details
    transaction_amounts = np.random.gamma(2, 500, n_samples).clip(10, 10000)
    days_overdue = np.random.exponential(10, n_samples).clip(0, 90).astype(int)
    
    # Categorical: expense categories
    categories = ['Travel', 'Utilities', 'Office Supplies', 'Professional Services', 
                  'Marketing', 'Rent', 'Payroll', 'Insurance']
    expense_category = np.random.choice(categories, n_samples, p=[0.15, 0.15, 0.15, 0.12, 
                                                                   0.12, 0.12, 0.10, 0.09])
    
    # Generate fraud flag (supervised target) - more likely with certain patterns
    fraud_prob = (1 / (1 + np.exp(-(-4 + 0.001 * transaction_amounts + 
                                   0.05 * days_overdue - 0.00001 * annual_incomes))))
    is_fraud = np.random.binomial(1, fraud_prob.clip(0, 1))
    
    # Client segments (for unsupervised evaluation)
    segment_scores = np.random.normal(0, 1, n_samples)
    client_segment = pd.cut(segment_scores, bins=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    # Dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': range(1, n_samples + 1),
        'transaction_date': dates,
        'client_age': client_ages,
        'annual_income': annual_incomes.astype(int),
        'transaction_amount': transaction_amounts.round(2),
        'days_overdue': days_overdue,
        'expense_category': expense_category,
        'is_fraud': is_fraud.astype(bool),
        'client_segment': client_segment
    })
    
    # Add some realistic correlations
    df['is_high_value'] = (df['transaction_amount'] > 5000).astype(bool)
    
    return df

if __name__ == "__main__":
    # Generate and save sample data
    df = generate_accounting_data(2000)
    df.to_csv('sample_accounting_data.csv', index=False)
    print("✅ Sample data generated and saved to 'sample_accounting_data.csv'")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    print(f"🎯 Fraud rate: {df['is_fraud'].mean():.2%}")
