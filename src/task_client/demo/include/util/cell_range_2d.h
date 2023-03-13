#ifndef SEMANTIC_MAP_UTIL_CELL_RANGE_2D_H
#define SEMANTIC_MAP_UTIL_CELL_RANGE_2D_H

#include <algorithm>
#include <limits>
#include <utility>

namespace SemanticMap {

    /**
     * 2d cell range
     */
    struct CellRange2d {
        int min_x_index_{std::numeric_limits<int>::max()};
        int max_x_index_{std::numeric_limits<int>::min()};
        int min_y_index_{std::numeric_limits<int>::max()};
        int max_y_index_{std::numeric_limits<int>::min()};

        void addPoint(const std::pair<int, int> occupied_cell) {
            const auto &[x_index, y_index] = occupied_cell;
            min_x_index_ = std::min(min_x_index_, x_index);
            max_x_index_ = std::max(max_x_index_, x_index);
            min_y_index_ = std::min(min_y_index_, y_index);
            max_y_index_ = std::max(max_y_index_, y_index);
        }

        void mergeRange(const CellRange2d &another_range) {
            min_x_index_ = std::min(min_x_index_, another_range.min_x_index_);
            max_x_index_ = std::max(max_x_index_, another_range.max_x_index_);
            min_y_index_ = std::min(min_y_index_, another_range.min_y_index_);
            max_y_index_ = std::max(max_y_index_, another_range.max_y_index_);
        }

        int get_x_range() const {
            return max_x_index_ - min_x_index_ + 1;
        }

        int get_y_range() const {
            return max_y_index_ - min_y_index_ + 1;
        }
    };

}

#endif // SEMANTIC_MAP_UTIL_CELL_RANGE_2D_H
