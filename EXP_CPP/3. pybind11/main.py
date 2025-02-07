# main.py
import csv
import time
import schedule  # The compiled pybind11 module
# or from . import schedule if it's in a package

def main():
    input_csv = "testing.csv"
    output_csv = "cpp_inprocess_output.csv"

    data = []
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    t0 = time.time()
    result_rows = []
    for row in data:
        unique_id = row["unique_reference_id"]
        issue_date = row["issue_date"]
        maturity_date = row["maturity_date"]
        freq = int(row["payment_frequency"])
        freq_unit = row["payment_frequency_units"]
        stub_date = row["stub_date"]

        # Call the C++ function directly
        schedule_pairs = schedule.generate_schedule(issue_date,
                                                   maturity_date,
                                                   freq,
                                                   freq_unit,
                                                   stub_date)
        # schedule_pairs is a list of (begin_str, end_str)
        for (b, e) in schedule_pairs:
            result_rows.append((unique_id, b, e))

    total_time = time.time() - t0
    print(f"C++ in-process schedule generation took {total_time:.9f} seconds.")

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unique_reference_id", "Begining_Date", "Ending_Date"])
        writer.writerows(result_rows)


if __name__ == "__main__":
    main()
