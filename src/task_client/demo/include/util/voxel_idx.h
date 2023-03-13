#ifndef SEMANTIC_MAP_UTIL_VOXEL_IDX_H
#define SEMANTIC_MAP_UTIL_VOXEL_IDX_H

#include <Eigen/Core>

namespace SemanticMap {

    using VoxelIdx = Eigen::Vector3i;

    struct voxelIdxHash {
        size_t operator()(const VoxelIdx &key) const {
            return size_t((key.x() * 73856093) ^ (key.y() * 471943) ^ (key.z() * 83492791)) % 10000000;
        }
    };

    struct pair_hash {
        template<typename T1, typename T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ h2;
        }
    };

}

#endif // SEMANTIC_MAP_UTIL_VOXEL_IDX_H
