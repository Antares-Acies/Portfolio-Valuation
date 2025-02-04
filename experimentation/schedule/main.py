# main.py
import csv
import time
import subprocess
import schedule  # Our schedule.py
from datetime import datetime

def python_schedule_for_row(row):
    """Generate the schedule using the Python approach."""
    unique_id = row["unique_reference_id"]
    issue_date = row["issue_date"]
    maturity_date = row["maturity_date"]
    payment_freq = row["payment_frequency"]
    payment_freq_unit = row["payment_frequency_units"]
    stub_date = row["stub_date"]
    # ignoring payment_amount in this snippet, but you can store/use it if needed

    begin_arr, end_arr = schedule.schedule_generation(
        issue_date, maturity_date, payment_freq, payment_freq_unit, stub_date
    )

    # Convert to rows like [unique_id, begin, end] for each step
    output_rows = []
    for b, e in zip(begin_arr, end_arr):
        out_b = b.strftime("%d-%m-%Y")
        out_e = e.strftime("%d-%m-%Y")
        output_rows.append([unique_id, out_b, out_e])
    return output_rows

def cpp_schedule_for_row(row, exe_path="./schedule.exe"):
    """Generate the schedule using the C++ approach via subprocess."""
    unique_id = row["unique_reference_id"]
    issue_date = row["issue_date"]
    maturity_date = row["maturity_date"]
    payment_freq = row["payment_frequency"]
    payment_freq_unit = row["payment_frequency_units"]
    stub_date = row["stub_date"]

    # Build argument list
    # schedule.exe <issue_date> <maturity_date> <freq> <freq_unit> [stub_date]
    cmd = [exe_path, issue_date, maturity_date, payment_freq, payment_freq_unit]
    if stub_date not in ["NaT", "-", "", "None"]:
        cmd.append(stub_date)

    # Run and capture stdout
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in C++ schedule for {unique_id}:", result.stderr)
        return []

    lines = result.stdout.strip().split("\n")
    output_rows = []
    for line in lines:
        if line.strip():
            b, e = line.split(",")
            output_rows.append([unique_id, b, e])
    return output_rows


def main():
    # 1. Read testing.csv
    input_csv = "testing.csv"
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)  # store all in memory for simplicity

    print(f"len of row in csv is {len(rows)}")

    # 2. Python Approach
    python_output = []
    t0 = time.time()
    for row in rows:
        out_rows = python_schedule_for_row(row)
        python_output.extend(out_rows)
    python_time = time.time() - t0

    # 3. Write python_output.csv
    with open("python_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_reference_id", "Begining_Date", "Ending_Date"])
        writer.writerows(python_output)

    print(f"Python schedule generation took {python_time:.9f} seconds.")
    # 4. C++ Approach
    cpp_output = []
    t0 = time.time()
    for row in rows:
        out_rows = cpp_schedule_for_row(row, exe_path="./schedule.exe")
        cpp_output.extend(out_rows)
    cpp_time = time.time() - t0

    # 5. Write cpp_output.csv
    with open("cpp_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_reference_id", "Begining_Date", "Ending_Date"])
        writer.writerows(cpp_output)

    # 6. Compare performance
    print(f"C++ schedule generation took {cpp_time:.9f} seconds.")

    faster = "Python" if python_time < cpp_time else "C++"
    factor = (cpp_time / python_time) if python_time != 0 else float('inf')
    if faster == "Python":
        factor = (python_time / cpp_time) if cpp_time != 0 else float('inf')

    print(f"{faster} is faster by factor of {factor:.7f}" if factor != float('inf') else "One approach took 0 time?")

    # OPTIONAL: Compare final CSVs to ensure they match line-by-line
    # In practice, you'd parse them and confirm they produce identical schedules.

if __name__ == "__main__":
    main()
