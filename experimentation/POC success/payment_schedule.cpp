#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <stdexcept>

// Parse a date string in "YYYY-MM-DD" format into std::time_t
std::time_t parse_date(const std::string &date_str) {
    std::tm tm = {};
    std::istringstream ss(date_str);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse date: " + date_str);
    }
    return std::mktime(&tm);
}

int main(int argc, char* argv[]) {
    // Expecting 5 arguments:
    // 1) start_date_str ("YYYY-MM-DD")
    // 2) end_date_str   ("YYYY-MM-DD")
    // 3) payment_frequency (int)
    // 4) payment_frequency_unit ("days"/"weeks"/"months"/"years")
    // 5) payment_amount (double)
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " START_DATE END_DATE FREQ UNIT AMOUNT\n";
        std::cerr << "Example: " << argv[0] << " 2023-01-01 2023-12-31 1 months 1000.0\n";
        return 1;
    }

    std::string start_date_str = argv[1];
    std::string end_date_str = argv[2];
    int payment_frequency = std::stoi(argv[3]);
    std::string payment_frequency_unit = argv[4];
    double payment_amount = std::stod(argv[5]);

    // Convert dates to std::time_t
    std::time_t start = parse_date(start_date_str);
    std::time_t end   = parse_date(end_date_str);

    const int SECONDS_PER_DAY = 86400;
    std::time_t current = start;

    // Generate and print the schedule lines
    while (current <= end) {
        std::tm* tm_ptr = std::localtime(&current);
        char buffer[11]; // enough for "YYYY-MM-DD"
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", tm_ptr);

        // Print date and amount
        std::cout << buffer << "," << payment_amount << "\n";

        // Increment current by payment_frequency based on unit
        if (payment_frequency_unit == "days") {
            current += payment_frequency * SECONDS_PER_DAY;
        } else if (payment_frequency_unit == "weeks") {
            current += payment_frequency * 7 * SECONDS_PER_DAY;
        } else if (payment_frequency_unit == "months") {
            // Approximate a month as 30 days
            current += payment_frequency * 30 * SECONDS_PER_DAY;
        } else if (payment_frequency_unit == "years") {
            // Approximate a year as 365 days
            current += payment_frequency * 365 * SECONDS_PER_DAY;
        } else {
            std::cerr << "Unsupported frequency unit: " << payment_frequency_unit << "\n";
            return 1;
        }
    }

    return 0;
}
