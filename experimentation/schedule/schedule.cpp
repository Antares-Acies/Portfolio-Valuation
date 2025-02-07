// schedule.cpp
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <ctime>
#include <stdexcept>

// A quick function to parse "DD-MM-YYYY"
std::tm parse_date(const std::string &date_str) {
    std::tm tm = {};
    std::istringstream ss(date_str);
    ss >> std::get_time(&tm, "%d-%m-%Y");
    if (ss.fail()) {
        // We'll treat invalid date as 0 time
        // or you can throw an error
        throw std::runtime_error("Invalid date: " + date_str);
    }
    return tm;
}

// A helper to convert std::tm to time_t
time_t to_time_t(std::tm &tm_struct) {
    // On Windows, tm_struct is in local time, so be cautious. For a quick test, this is fine.
    return std::mktime(&tm_struct);
}

// Add months or years in a simplified manner
// For real code, you might handle day-of-month edge cases.
std::tm add_months(std::tm t, int months) {
    t.tm_mon += months; // increment months
    // Let mktime recalc day/year
    std::mktime(&t);
    return t;
}
std::tm add_years(std::tm t, int years) {
    t.tm_year += years; 
    std::mktime(&t);
    return t;
}

int main(int argc, char* argv[]) {
    // Usage: schedule.exe <issue_date> <maturity_date> <payment_frequency> <payment_frequency_unit> <stub_date?>
    // We'll expect either 5 or 6 arguments (stub date is optional).
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <issue_date> <maturity_date> <payment_frequency> <payment_frequency_unit> [stub_date]\n";
        return 1;
    }

    std::string issue_date_str = argv[1];
    std::string maturity_date_str = argv[2];
    int frequency = std::stoi(argv[3]);
    std::string freq_unit = argv[4];
    std::string stub_date_str = (argc == 6) ? argv[5] : "";

    // parse dates
    std::tm issue_tm, maturity_tm, stub_tm;
    try {
        issue_tm = parse_date(issue_date_str);
        maturity_tm = parse_date(maturity_date_str);
        if (!stub_date_str.empty()) {
            stub_tm = parse_date(stub_date_str);
        }
    } catch (const std::exception &ex) {
        std::cerr << "Date parsing error: " << ex.what() << "\n";
        return 1;
    }

    time_t issue_tt = to_time_t(issue_tm);
    time_t maturity_tt = to_time_t(maturity_tm);
    time_t stub_tt = (!stub_date_str.empty()) ? to_time_t(stub_tm) : 0;

    // For each "period," we print "begin_date,end_date"
    // We'll implement a naive approach similar to your Python logic

    auto generate_range = [&](time_t start_t, time_t end_t, int freq, const std::string &u) {
        std::vector<std::pair<time_t, time_t>> schedule;
        time_t current = start_t;
        while (current < end_t) {
            // Convert time_t -> std::tm
            std::tm current_tm = *std::localtime(&current);

            // Compute next
            std::tm next_tm = current_tm;
            if (u == "D") {
                next_tm.tm_mday += freq; 
                std::mktime(&next_tm);
            } else if (u == "M") {
                next_tm = add_months(next_tm, freq);
            } else if (u == "Q") {
                next_tm = add_months(next_tm, 3 * freq);
            } else if (u == "Y") {
                next_tm = add_years(next_tm, freq);
            } else {
                std::cerr << "Unsupported frequency unit: " << u << "\n";
                break;
            }
            time_t next_t = std::mktime(&next_tm);

            schedule.push_back({current, next_t});
            current = next_t;
            if (current >= end_t) break;
        }
        return schedule;
    };

    auto print_schedule = [&](const std::vector<std::pair<time_t, time_t>> &sch) {
        for (auto &pair : sch) {
            std::tm b = *std::localtime(&pair.first);
            std::tm e = *std::localtime(&pair.second);

            char buf_b[11];
            char buf_e[11];
            std::strftime(buf_b, sizeof(buf_b), "%d-%m-%Y", &b);
            std::strftime(buf_e, sizeof(buf_e), "%d-%m-%Y", &e);
            std::cout << buf_b << "," << buf_e << "\n";
        }
    };

    // If we have a stub date, we do two passes:
    if (!stub_date_str.empty()) {
        auto s1 = generate_range(issue_tt, stub_tt, frequency, freq_unit);
        auto s2 = generate_range(stub_tt, maturity_tt, frequency, freq_unit);

        print_schedule(s1);
        print_schedule(s2);
    } else {
        auto s = generate_range(issue_tt, maturity_tt, frequency, freq_unit);
        print_schedule(s);
    }

    return 0;
}
