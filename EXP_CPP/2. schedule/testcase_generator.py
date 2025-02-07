import csv
import random
from datetime import datetime, timedelta

# Number of data rows to generate (excluding the header)
num_rows = 500000

# Fixed reporting_date (this value remains constant for every row)
reporting_date_str = "31-07-2024"
reporting_date_dt = datetime.strptime(reporting_date_str, "%d-%m-%Y")

# Define a minimum date for issue_date so that issue_date < reporting_date.
# You can adjust this starting date as needed.
issue_date_min_dt = datetime(2020, 1, 1)
# Calculate the maximum number of days we can add to issue_date_min_dt so that the result is still before reporting_date.
max_issue_delta = (reporting_date_dt - issue_date_min_dt).days - 1

# Open (or create) the CSV file for writing
with open("testing.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header row
    writer.writerow([
        "unique_reference_id",
        "reporting_date",
        "issue_date",
        "payment_frequency",
        "payment_frequency_units",
        "stub_date",
        "maturity_date",
        "payment_amount"
    ])
    
    i=0
    for _ in range(num_rows):
        print(f"{i}")
        i+=1
        # Generate a unique_reference_id of the form "SPCO" followed by 7 digits.
        unique_reference_id = "SPCO" + str(random.randint(0, 9999999)).zfill(7)
        
        # Generate an issue_date that is strictly before reporting_date.
        # We randomly choose a number of days to add to the minimum date.
        delta_days = random.randint(0, max_issue_delta)
        issue_date_dt = issue_date_min_dt + timedelta(days=delta_days)
        issue_date_str = issue_date_dt.strftime("%d-%m-%Y")
        
        # Generate a payment_frequency (any positive integer, here between 1 and 12)
        payment_frequency = random.randint(1, 12)
        
        # Generate a payment_frequency_units value from the allowed set: D, M, Y, Q.
        payment_frequency_units = random.choice(["D", "M", "Y", "Q"])
        
        # Generate a maturity_date that is always after reporting_date.
        # Here we add a random number of days (between 1 and 1000) to the reporting_date.
        maturity_delta_days = random.randint(1, 1000)
        maturity_date_dt = reporting_date_dt + timedelta(days=maturity_delta_days)
        maturity_date_str = maturity_date_dt.strftime("%d-%m-%Y")
        
        # For stub_date, randomly decide to either leave it as "NaT"
        # or generate a date that is strictly before reporting_date.
        if random.random() < 0.5:
            stub_date_str = "NaT"
        else:
            # Generate stub_date between issue_date and reporting_date.
            # Calculate how many days lie between issue_date and reporting_date.
            days_between = (reporting_date_dt - issue_date_dt).days - 1
            if days_between > 0:
                stub_delta_days = random.randint(0, days_between)
                stub_date_dt = issue_date_dt + timedelta(days=stub_delta_days)
                stub_date_str = stub_date_dt.strftime("%d-%m-%Y")
            else:
                stub_date_str = "NaT"
        
        # Generate a payment_amount (a random float between 10.0 and 10,000.0)
        payment_amount = round(random.uniform(10.0, 10000.0), 2)
        
        # Write the generated row to the CSV file
        writer.writerow([
            unique_reference_id,
            reporting_date_str,
            issue_date_str,
            payment_frequency,
            payment_frequency_units,
            stub_date_str,
            maturity_date_str,
            payment_amount
        ])

print(f"CSV file 'testing.csv' generated with {num_rows} rows.")
