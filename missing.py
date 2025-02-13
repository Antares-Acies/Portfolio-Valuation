import os
import pandas as pd

# Define paths
cashflow_folder = r'C:\Users\AnuragSinha\OneDrive - ACIES\Desktop\PORT\Portfolio-Valuation\output\Cashflow_Engine_Outputs\Cashflow'
position_data_path = r'C:\Users\AnuragSinha\OneDrive - ACIES\Desktop\PORT\Portfolio-Valuation\Portfolio Valuation Data\position_data.csv'
output_csv_path = r'C:\Users\AnuragSinha\OneDrive - ACIES\Desktop\PORT\Portfolio-Valuation\failed_drop_position_data.csv'

# Read and concatenate all CSV files in the cashflow folder
cashflow_files = [os.path.join(cashflow_folder, file) for file in os.listdir(cashflow_folder) if file.endswith('.csv')]
combined_cashflow_data = pd.concat([pd.read_csv(file) for file in cashflow_files])

# Read position data
position_data = pd.read_csv(position_data_path)
# Find position_ids present in position_data but not in combined_cashflow_data
missing_positions = position_data[~position_data['position_id'].isin(combined_cashflow_data['position_id'])]

# Save the result to a new CSV file
missing_positions.to_csv(output_csv_path, index=False)

print(f"CSV file with missing position data has been created: {output_csv_path}")
