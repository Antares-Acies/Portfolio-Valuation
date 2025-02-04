# main.py
import subprocess

def main():
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    freq = "1"
    unit = "months"
    amount = "1000.0"

    # Run the payment_schedule.exe with command-line arguments
    # The subprocess will return the schedule lines via stdout
    cmd = [
        "./payment_schedule.exe",  # use "./" on Windows if it's in the same folder
        start_date,
        end_date,
        freq,
        unit,
        amount
    ]

    try:
        # capture_output=True or stdout=subprocess.PIPE to get the output in Python
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing payment_schedule.exe:", e)
        return

    # result.stdout now contains lines like "YYYY-MM-DD,<amount>\n"
    schedule_lines = result.stdout.strip().split("\n")

    # Parse each line if you want to convert into Python objects
    schedule = []
    for line in schedule_lines:
        if line.strip():
            date_str, amt_str = line.split(",")
            schedule.append({"date": date_str, "amount": float(amt_str)})

    # Print or do whatever you want with schedule
    for entry in schedule:
        print(entry)

if __name__ == "__main__":
    main()
