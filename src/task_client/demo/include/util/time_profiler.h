#ifndef SEMANTIC_MAP_UTIL_TIME_PROFILER_H
#define SEMANTIC_MAP_UTIL_TIME_PROFILER_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace SemanticMap {
    class TimeProfiler {
    public:
        TimeProfiler() {
            start_time_point_ = std::chrono::steady_clock::now();
        }

        void recordTimePoint(const std::string &label) {
            const auto time_point = std::chrono::steady_clock::now();
            label_to_time_points.emplace_back(std::make_pair(label, time_point));
        }

        void printResult() {
            const auto &end_time = label_to_time_points.at(label_to_time_points.size() - 1).second;
            std::cout << std::endl << std::string(73, '-') << std::endl;
            std::cout << "|" << std::setw(60) << "overall time: " << "|"
                      << std::to_string(std::chrono::duration<double>(end_time - start_time_point_).count())
                      << "s |" << std::endl;
            auto last_time_point = start_time_point_;
            for (const auto &[label, time_point]: label_to_time_points) {
                std::cout << "|" << std::setw(60) << label << "|"
                          << std::to_string(std::chrono::duration<double>(time_point - last_time_point).count())
                          << "s |" << std::endl;
                last_time_point = time_point;
            }
            std::cout << std::string(73, '-') << std::endl << std::endl;
        }

    private:
        std::chrono::steady_clock::time_point start_time_point_;

        std::vector<std::pair<std::string, std::chrono::steady_clock::time_point>> label_to_time_points;
    };

}

#endif // SEMANTIC_MAP_UTIL_TIME_PROFILER_H
