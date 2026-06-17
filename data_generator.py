import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def generate_accounting_data(n_samples=2000, fraud_rate=0.10, seed=42):
    """
    Generate synthetic accounting data with realistic fraud patterns.
    
    Parameters:
    - n_samples: Total number of transactions
    - fraud_rate: Target fraud rate (default 10% - realistic for demonstrations)
    - seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # ============================================
    # 📊 GENERATE CLIENT DATA
    # ============================================
    
    # Client demographics
    client_ages = np.random.randint(22, 70, n_samples)
    annual_incomes = np.random.normal(75000, 30000, n_samples).clip(20000, 250000)
    
    # ============================================
    # 💰 GENERATE TRANSACTIONS
    # ============================================
    
    # Transaction amounts (gamma distribution for realistic skew)
    transaction_amounts = np.random.gamma(2, 500, n_samples).clip(10, 10000)
    
    # Days overdue (exponential distribution)
    days_overdue = np.random.exponential(10, n_samples).clip(0, 90).astype(int)
    
    # Expense categories with realistic probabilities
    categories = ['Travel', 'Utilities', 'Office Supplies', 'Professional Services', 
                  'Marketing', 'Rent', 'Payroll', 'Insurance']
    expense_category = np.random.choice(categories, n_samples, p=[0.15, 0.15, 0.15, 0.12, 
                                                                   0.12, 0.12, 0.10, 0.09])
    
    # ============================================
    # 🎯 GENERATE FRAUD WITH REALISTIC PATTERNS
    # ============================================
    
    # Base fraud probability (logistic function with multiple factors)
    # Higher amounts + more overdue = higher fraud probability
    fraud_prob_raw = (
        0.001 * transaction_amounts +           # High amounts increase risk
        0.03 * days_overdue +                   # Delayed payments increase risk
        -0.00001 * annual_incomes +             # Higher income = slightly lower risk
        0.5 * (transaction_amounts > 5000)      # Very high amounts are risky
    )
    
    # Scale to achieve target fraud rate
    fraud_prob_raw = fraud_prob_raw - fraud_prob_raw.mean() + np.log(fraud_rate / (1 - fraud_rate))
    fraud_prob = 1 / (1 + np.exp(-fraud_prob_raw))
    fraud_prob = np.clip(fraud_prob, 0.01, 0.99)  # Keep probabilities reasonable
    
    # Generate fraud labels
    is_fraud = np.random.binomial(1, fraud_prob)
    
    # ============================================
    # 📊 ENSURE MINIMUM FRAUD CASES
    # ============================================
    
    # If too few fraud cases, force some high-risk transactions to be fraud
    fraud_count = is_fraud.sum()
    target_fraud_count = int(n_samples * fraud_rate)
    
    if fraud_count < target_fraud_count * 0.5:  # If less than half of target
        # Find high-risk transactions that weren't flagged as fraud
        high_risk_indices = np.argsort(fraud_prob)[::-1]  # Sort by probability descending
        fraud_indices = np.where(is_fraud == 1)[0]
        
        # Additional fraud indices
        additional_needed = target_fraud_count - fraud_count
        additional_indices = [idx for idx in high_risk_indices if idx not in fraud_indices][:additional_needed]
        
        # Set them as fraud
        for idx in additional_indices:
            is_fraud[idx] = 1
        
        fraud_count = is_fraud.sum()
    
    # ============================================
    # 📊 CLIENT SEGMENTS (for unsupervised learning)
    # ============================================
    
    segment_scores = np.random.normal(0, 1, n_samples)
    client_segment = pd.cut(segment_scores, bins=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    # ============================================
    # 📅 DATES
    # ============================================
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
    # ============================================
    # 📋 CREATE DATAFRAME
    # ============================================
    
    df = pd.DataFrame({
        'transaction_id': range(1, n_samples + 1),
        'transaction_date': dates,
        'client_age': client_ages,
        'annual_income': annual_incomes.astype(int),
        'transaction_amount': transaction_amounts.round(2),
        'days_overdue': days_overdue,
        'expense_category': expense_category,
        'is_fraud': is_fraud.astype(bool),
        'client_segment': client_segment,
        'fraud_probability': fraud_prob.round(3)  # Store for reference
    })
    
    # Add some realistic correlations
    df['is_high_value'] = (df['transaction_amount'] > 5000).astype(bool)
    
    # ============================================
    # 📊 SUMMARY STATISTICS
    # ============================================
    
    fraud_rate_actual = df['is_fraud'].mean()
    
    print(f"""
    📊 Data Generation Summary:
    ============================
    Total Samples: {len(df)}
    Fraud Cases: {df['is_fraud'].sum()}
    Fraud Rate: {fraud_rate_actual:.2%}
    Target Rate: {fraud_rate:.2%}
    
    Expense Categories:
    {df['expense_category'].value_counts().to_string()}
    
    Amount Statistics:
    Mean: ${df['transaction_amount'].mean():.2f}
    Max: ${df['transaction_amount'].max():.2f}
    """)
    
    return df

# ============================================
# 🚀 BALANCED DATA GENERATOR (Guaranteed Balance)
# ============================================

def generate_balanced_accounting_data(n_samples=2000, fraud_rate=0.10, seed=42):
    """
    Generate balanced data with guaranteed minimum fraud cases.
    Useful when you want to ensure enough fraud examples for training.
    """
    # First pass - generate data
    df = generate_accounting_data(n_samples, fraud_rate, seed)
    
    # Check fraud count
    fraud_count = df['is_fraud'].sum()
    required_fraud = max(10, int(n_samples * fraud_rate))
    
    if fraud_count < required_fraud:
        st.warning(f"⚠️ Only {fraud_count} fraud cases found. Generating more balanced data...")
        
        # Generate new data with higher fraud rate
        # Increase fraud rate progressively until we get enough
        new_fraud_rate = fraud_rate
        max_attempts = 5
        
        for attempt in range(max_attempts):
            new_fraud_rate = min(new_fraud_rate * 1.5, 0.5)  # Cap at 50%
            df_new = generate_accounting_data(n_samples, new_fraud_rate, seed + attempt + 1)
            fraud_count_new = df_new['is_fraud'].sum()
            
            if fraud_count_new >= required_fraud:
                df = df_new
                st.success(f"✅ Generated {fraud_count_new} fraud cases ({fraud_count_new/n_samples:.2%})")
                break
        
        # If still not enough, force balance
        if df['is_fraud'].sum() < required_fraud:
            # Take the best attempt and force some cases to be fraud
            fraud_indices = np.where(df['is_fraud'] == False)[0]
            additional_needed = required_fraud - df['is_fraud'].sum()
            
            # Sort by fraud probability
            fraud_probs = df.loc[fraud_indices, 'fraud_probability'].values
            top_indices = fraud_indices[np.argsort(fraud_probs)[-additional_needed:]]
            
            df.loc[top_indices, 'is_fraud'] = True
            st.info(f"🔄 Forced {additional_needed} additional high-risk cases to be fraud")
    
    return df

# ============================================
# 📥 MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Generate sample data
    df = generate_balanced_accounting_data(2000, fraud_rate=0.10)
    df.to_csv('sample_accounting_data.csv', index=False)
    print("✅ Sample data generated and saved to 'sample_accounting_data.csv'")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    print(f"🎯 Fraud rate: {df['is_fraud'].mean():.2%}")
