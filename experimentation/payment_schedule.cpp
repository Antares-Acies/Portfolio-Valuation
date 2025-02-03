// payment_schedule.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace py = pybind11;

// A helper function to parse a date string in "YYYY-MM-DD" format into time_t.
std::time_t parse_date(const std::string &date_str) {
    std::tm tm = {};
    std::istringstream ss(date_str);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    if (ss.fail()) {
        throw std::runtime_error("Failed to parse date: " + date_str);
    }
    return std::mktime(&tm);
}

// Generate a payment schedule from start_date to end_date using the given frequency.
std::vector<std::string> generate_payment_schedule(const std::string& start_date_str,
                                                     const std::string& end_date_str,
                                                     int payment_frequency,
                                                     const std::string& payment_frequency_unit,
                                                     double payment_amount) {
    // Parse dates
    std::time_t start = parse_date(start_date_str);
    std::time_t end = parse_date(end_date_str);

    std::vector<std::string> schedule;
    const int SECONDS_PER_DAY = 86400;
    std::time_t current = start;

    // Simple date incrementation:
    while (current <= end) {
        std::tm *tm_ptr = std::localtime(&current);
        char buffer[11]; // "YYYY-MM-DD" + null terminator
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", tm_ptr);

        // Create a simple string representing the payment (date + amount)
        schedule.push_back(std::string(buffer) + " : " + std::to_string(payment_amount));

        // Increment the current date based on the frequency unit.
        if (payment_frequency_unit == "days") {
            current += payment_frequency * SECONDS_PER_DAY;
        } else if (payment_frequency_unit == "weeks") {
            current += payment_frequency * 7 * SECONDS_PER_DAY;
        } else if (payment_frequency_unit == "months") {
            // For simplicity, approximate a month as 30 days.
            current += payment_frequency * 30 * SECONDS_PER_DAY;
        } else if (payment_frequency_unit == "years") {
            // Approximate a year as 365 days.
            current += payment_frequency * 365 * SECONDS_PER_DAY;
        } else {
            throw std::runtime_error("Unsupported payment frequency unit: " + payment_frequency_unit);
        }
    }
    return schedule;
}

PYBIND11_MODULE(payment_schedule, m) {
    m.doc() = "Module for generating payment schedules using C++ with pybind11";
    m.def("generate_payment_schedule", &generate_payment_schedule, 
          "Generate payment schedule",
          py::arg("start_date_str"),
          py::arg("end_date_str"),
          py::arg("payment_frequency"),
          py::arg("payment_frequency_unit"),
          py::arg("payment_amount"));
}
