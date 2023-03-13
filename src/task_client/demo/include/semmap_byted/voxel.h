#ifndef SEMANTIC_MAP_VOXEL_HPP
#define SEMANTIC_MAP_VOXEL_HPP

#include "util.h"

#include <Eigen/Core>
#include <pcl/point_types.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>



namespace SemanticMap {

    class Voxel {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        static constexpr std::size_t MAX_POINT_NUM_IN_VOXEL = 10;

        Voxel(Eigen::Vector3i voxel_idx, Eigen::Vector3f center, const float side_length)
                : voxel_idx_(std::move(voxel_idx)), center_(std::move(center)), side_length_(side_length) {};

        /**
         * add \p point into this voxel
         * @param[in] point                     the point to be added
         */
        void addPoint(const pcl::PointXYZ &point) {
//            assert(std::abs(point.x - center_.x()) < side_length_ / 2);
//            assert(std::abs(point.y - center_.y()) < side_length_ / 2);
//            assert(std::abs(point.z - center_.z()) < side_length_ / 2);
            next_input_point_index_ += 1;
            if (countPointNum() < MAX_POINT_NUM_IN_VOXEL) {
                points_.emplace_back(point);
            } else {
                points_.at(next_input_point_index_ % MAX_POINT_NUM_IN_VOXEL) = point;
            }
        };

        /**
         * check if the voxel is empty
         *
         * @return                              if the voxel is empty
         */
        bool empty() const {
            return points_.empty();
        };

        /**
         * count point num in this voxel
         *
         * @return                              point num in this voxel
         */
        std::size_t countPointNum() const {
            return points_.size();
        };

        /**
         * get voxel idx
         *
         * @return                              voxel idx
         */
        VoxelIdx getVoxelIdx() const {
            return voxel_idx_;
        }

        /**
         * get point with index as \p idx in this voxel
         *
         * @param[in] idx                       target point index
         * @return                              point with index as \p idx
         */
        pcl::PointXYZ getPointByIdx(const std::size_t idx) const {
            return points_.at(idx);
        };

        /**
         * distance information between a point and target point
         */
        struct PointDistanceInfo {

            PointDistanceInfo() = default;

            PointDistanceInfo(const double distance, const Voxel *i_vox_node, const int point_idx) :
                    distance_(distance), voxel_node_(i_vox_node), point_idx_(point_idx) {};

            /**
             * get corresponding point
             *
             * @return              corresponding point
             */
            pcl::PointXYZ getPoint() const {
                return voxel_node_->getPointByIdx(point_idx_);
            }

            /**
             * sort by distance
             */
            inline bool operator<(const PointDistanceInfo &rhs) const { return distance_ < rhs.distance_; }

        private:
            /// voxel of this point
            const Voxel *voxel_node_;

            /// index of this point in voxel
            int point_idx_;

            /// distance to target point
            double distance_;
        };

        /**
         * add at most \p k distance information between \p target_point and point in this voxel into \distance_infos.
         * save at most \p k information from this voxel and each distance no more than \p max_distance.
         *
         * @param[in, out] distance_infos       save distance information into this vector
         * @param[in] target_point              compute distances to this point
         * @param[in] k                         add at most k information from this voxel
         * @param[in] max_distance              every added distance should no more than max_distance
         * @return                              num of added information from this voxel
         */
        int updateDistanceInfoKNN(
                std::vector<PointDistanceInfo> &distance_infos, const pcl::PointXYZ &target_point, const int &k,
                const double &max_distance) const {
            std::size_t old_size = distance_infos.size();
            // compute distance infos of points in voxel
            for (const auto &point: points_) {
                const auto squared_distance = Eigen::Vector3f(
                        point.getVector3fMap() - target_point.getVector3fMap()).squaredNorm();
                const auto info = PointDistanceInfo(squared_distance, this, &point - points_.data());
                if (squared_distance < max_distance * max_distance) {
                    distance_infos.emplace_back(info);
                }
            }
            // cull redundant infos
            if (old_size + k < distance_infos.size()) {
                std::nth_element(
                        distance_infos.begin() + old_size, distance_infos.begin() + old_size + k - 1,
                        distance_infos.end());
                distance_infos.resize(old_size + k);
            }
            return distance_infos.size();
        }

        /**
         * merge points from \p another_voxel into this voxel.
         *
         * @param[in] another_voxel             voxel to be merged into this voxel
         */
        void mergeVoxel(const Voxel &another_voxel) {
            for (const auto &point: another_voxel.points_) {
                addPoint(point);
            }
        }

    private:
        Voxel() = default;

        /// voxel index
        VoxelIdx voxel_idx_;

        /// center of this voxel grid, not center of points in this voxel
        Eigen::Vector3f center_;

        /// voxel side length
        float side_length_;

        /// points in voxel
        std::vector<pcl::PointXYZ> points_;

        /// index of new input point
        std::size_t next_input_point_index_{0};
    };

} // namespace SemanticMap

#endif// SEMANTIC_MAP_VOXEL_HPP
