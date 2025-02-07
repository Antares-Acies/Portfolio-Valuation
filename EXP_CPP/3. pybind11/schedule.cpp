// schedule.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // for automatic conversion of std::vector, etc.
#include <vector>
#include <string>
#include <stdexcept>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace py = pybind11;

// Helper: parse "DD-MM-YYYY" to std::tm
std::tm parse_date(const std::string &date_str) {
    std::tm tm = {};
    std::istringstream ss(date_str);
    ss >> std::get_time(&tm, "%d-%m-%Y");
    if (ss.fail()) {
        throw std::runtime_error("Invalid date: " + date_str);
    }
    return tm;
}

// Convert std::tm to time_t (local time). Adjust as needed for your timezone.
time_t to_time_t(std::tm &tm_struct) {
    return std::mktime(&tm_struct);
}

// Add months/years in a naive way.
std::tm add_months(std::tm t, int months) {
    t.tm_mon += months; 
    std::mktime(&t);
    return t;
}
std::tm add_years(std::tm t, int years) {
    t.tm_year += years; 
    std::mktime(&t);
    return t;
}

/**
 * generate_schedule:
 *   - issue_date_str: "DD-MM-YYYY"
 *   - maturity_date_str: "DD-MM-YYYY"
 *   - freq: integer frequency (e.g. 1, 3, etc.)
 *   - freq_unit: "D", "M", "Q", or "Y"
 *   - stub_date_str: optional "DD-MM-YYYY" or empty string
 *
 * Returns: std::vector<std::pair<std::string, std::string>>
 *          containing (begin_date, end_date) pairs as strings.
 */
std::vector<std::pair<std::string, std::string>> generate_schedule(
    const std::string &issue_date_str,
    const std::string &maturity_date_str,
    int freq,
    const std::string &freq_unit,
    const std::string &stub_date_str = ""
) {
    // Parse
    std::tm issue_tm = parse_date(issue_date_str);
    std::tm maturity_tm = parse_date(maturity_date_str);
    time_t issue_tt = to_time_t(issue_tm);
    time_t maturity_tt = to_time_t(maturity_tm);

    time_t stub_tt = 0;
    bool has_stub = !stub_date_str.empty() && stub_date_str != "NaT" && stub_date_str != "-";
    if (has_stub) {
        std::tm stub_tm = parse_date(stub_date_str);
        stub_tt = to_time_t(stub_tm);
    }

    // We'll store all pairs of (begin_date_str, end_date_str)
    std::vector<std::pair<std::string, std::string>> schedule;

    // Helper lambda to convert time_t -> "DD-MM-YYYY" string
    auto to_string_ddmmyyyy = [&](time_t t) {
        std::tm localt = *std::localtime(&t);
        char buf[11];
        std::strftime(buf, sizeof(buf), "%d-%m-%Y", &localt);
        return std::string(buf);
    };

    // A function to generate a range from start_tt to end_tt
    auto generate_range = [&](time_t start_tt, time_t end_tt) {
        std::vector<std::pair<std::string, std::string>> result;
        time_t current = start_tt;
        while (current < end_tt) {
            std::tm current_tm = *std::localtime(&current);

            // Compute next
            std::tm next_tm = current_tm;
            if (freq_unit == "D") {
                next_tm.tm_mday += freq;
                std::mktime(&next_tm);
            } else if (freq_unit == "M") {
                next_tm = add_months(next_tm, freq);
            } else if (freq_unit == "Q") {
                next_tm = add_months(next_tm, 3 * freq);
            } else if (freq_unit == "Y") {
                next_tm = add_years(next_tm, freq);
            } else {
                throw std::runtime_error("Unsupported frequency unit: " + freq_unit);
            }
            time_t next_tt = std::mktime(&next_tm);

            // Add pair
            result.push_back({ to_string_ddmmyyyy(current), to_string_ddmmyyyy(next_tt) });

            current = next_tt;
            if (current >= end_tt) break;
        }
        return result;
    };

    if (has_stub) {
        // 2 segments: issue->stub, stub->maturity
        auto seg1 = generate_range(issue_tt, stub_tt);
        auto seg2 = generate_range(stub_tt, maturity_tt);
        // Concatenate
        schedule.insert(schedule.end(), seg1.begin(), seg1.end());
        schedule.insert(schedule.end(), seg2.begin(), seg2.end());
    } else {
        // Single segment
        auto seg = generate_range(issue_tt, maturity_tt);
        schedule.insert(schedule.end(), seg.begin(), seg.end());
    }

    return schedule;
}

// The pybind11 module definition
PYBIND11_MODULE(schedule, m) {
    m.doc() = "Schedule generation module using C++ and pybind11";

    // Expose generate_schedule(...) to Python
    m.def("generate_schedule",
          &generate_schedule,
          py::arg("issue_date_str"),
          py::arg("maturity_date_str"),
          py::arg("freq"),
          py::arg("freq_unit"),
          py::arg("stub_date_str") = "",
          "Generate a schedule of date pairs (begin, end)."
    );
}
