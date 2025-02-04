# schedule.py

import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def parse_date(date_str):
    """Parse 'DD-MM-YYYY' into a Python datetime."""
    # Handle special cases like 'NaT' or '-'
    if date_str in ["NaT", "-", "None", "nan"]:
        return None
    return datetime.strptime(date_str, "%d-%m-%Y")

def date_range_custom(start_date, end_date, frequency, frequency_unit):
    """Generate arrays of begin/end dates between start_date and end_date,
       stepping by (frequency, frequency_unit)."""
    begin_dates = []
    end_dates = []
    current = start_date

    while current < end_date:
        begin_dates.append(current)
        if frequency_unit == 'D':
            next_date = current + timedelta(days=frequency)
        elif frequency_unit == 'M':
            next_date = current + relativedelta(months=frequency)
        elif frequency_unit == 'Q':
            # e.g., 3 months * frequency
            next_date = current + relativedelta(months=3*frequency)
        elif frequency_unit == 'Y':
            next_date = current + relativedelta(years=frequency)
        else:
            raise ValueError(f"Unsupported frequency unit: {frequency_unit}")

        end_dates.append(next_date)
        current = next_date

    return np.array(begin_dates), np.array(end_dates)

def schedule_generation(issue_date_str, maturity_date_str,
                        payment_frequency, payment_frequency_unit,
                        stub_date_str=None):
    """Generate schedule arrays (begin/end)."""

    issue_date = parse_date(issue_date_str)
    maturity_date = parse_date(maturity_date_str)
    stub_date = parse_date(stub_date_str) if stub_date_str else None

    if issue_date is None or maturity_date is None:
        # In real code, handle or raise an error
        return [], []

    # Convert payment_frequency to integer for stepping
    freq = int(payment_frequency)

    # If no stub date
    if not stub_date:
        begin_arr, end_arr = date_range_custom(
            issue_date, maturity_date, freq, payment_frequency_unit
        )
    else:
        # Example approach: first range from issue_date to stub_date, then stub_date to maturity
        # This is just an example of stub handling. Adjust logic to your real needs.
        b0, e0 = date_range_custom(issue_date, stub_date, freq, payment_frequency_unit)
        b1, e1 = date_range_custom(stub_date, maturity_date, freq, payment_frequency_unit)

        # Concatenate
        begin_arr = np.concatenate((b0, b1))
        end_arr = np.concatenate((e0, e1))

    return begin_arr, end_arr
