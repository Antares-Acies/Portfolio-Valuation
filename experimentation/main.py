# main.py

import build_ext

def main():
    # Compile and import the latest version of the C++ module.
    payment_schedule = build_ext.build_and_import()
    
    # Example parameters for generating a payment schedule.
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    payment_frequency = 1
    payment_frequency_unit = "months"
    payment_amount = 1000.0
    
    # Call the C++ function via the imported module.
    schedule = payment_schedule.generate_payment_schedule(
        start_date, end_date, payment_frequency, payment_frequency_unit, payment_amount
    )
    
    # Print out the payment schedule.
    for payment in schedule:
        print(payment)

if __name__ == "__main__":
    main()
