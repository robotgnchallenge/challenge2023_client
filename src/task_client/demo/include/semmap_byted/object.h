#ifndef SEMANTIC_MAP_OBJECT_HPP
#define SEMANTIC_MAP_OBJECT_HPP

#include "util.h"
#include "voxel.h"

#include <Eigen/Core>
#include <pcl/io/auto_io.h>
#include <pcl/make_shared.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/console.h>

#include <algorithm>
#include <execution>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static int roundToNearest(const float f) {
    return static_cast<int>(f + (f > 0 ? 0.5f :-0.5f));
}

namespace SemanticMap {

    class Object {
    public:
        using PointVector = std::vector<pcl::PointXYZ>;
        using PointDistanceInfo = Voxel::PointDistanceInfo;

        /// confidence of object constructed from point cloud
        static constexpr float CONFIDENCE_FROM_POINT_CLOUD = 1.0f;

        /// confidence of object read from file
        static constexpr float CONFIDENCE_FROM_FILE = 0.8f;

        /// max feature num of an object
        static constexpr int MAX_FEATURE_NUM = 3;

        struct Attribute {
            /// neighbour search policy for KNN
            enum class NeighbourSearchPolicy {
                CENTER, NEARBY6, NEARBY18, NEARBY26
            };

            Attribute(const float voxel_side_length, const size_t capacity) :
                    voxel_side_length_(voxel_side_length), capacity_(capacity) {}

            /// voxel side length
            float voxel_side_length_;

            /// KNN search policy for voxels in object
            NeighbourSearchPolicy neighbour_search_policy_{NeighbourSearchPolicy::NEARBY6};

            /// max voxel num in an object
            std::size_t capacity_;
        };

        explicit Object(const ObjectType object_type = ObjectType::UNKNOWN, const uint16_t object_id = -1,
                        const float voxel_side_length = 0.05, const float confidence = CONFIDENCE_FROM_POINT_CLOUD,
                        const std::size_t capacity = 10000000)
                : object_type_(object_type), confidence_(confidence), object_id_(object_id),
                  attribute_(voxel_side_length, capacity) {
            // generate nearby voxels
            switch (attribute_.neighbour_search_policy_) {
                case Attribute::NeighbourSearchPolicy::CENTER:
                    neighbour_voxel_idx_offsets_ = {VoxelIdx::Zero()};
                    break;
                case Attribute::NeighbourSearchPolicy::NEARBY6:
                    neighbour_voxel_idx_offsets_ = {
                            VoxelIdx(0, 0, 0), VoxelIdx(-1, 0, 0), VoxelIdx(1, 0, 0), VoxelIdx(0, 1, 0),
                            VoxelIdx(0, -1, 0), VoxelIdx(0, 0, -1),
                            VoxelIdx(0, 0, 1)};
                    break;
                case Attribute::NeighbourSearchPolicy::NEARBY18:
                    neighbour_voxel_idx_offsets_ = {
                            VoxelIdx(0, 0, 0), VoxelIdx(-1, 0, 0), VoxelIdx(1, 0, 0), VoxelIdx(0, 1, 0),
                            VoxelIdx(0, -1, 0), VoxelIdx(0, 0, -1),
                            VoxelIdx(0, 0, 1), VoxelIdx(1, 1, 0), VoxelIdx(-1, 1, 0), VoxelIdx(1, -1, 0),
                            VoxelIdx(-1, -1, 0),
                            VoxelIdx(1, 0, 1), VoxelIdx(-1, 0, 1), VoxelIdx(1, 0, -1), VoxelIdx(-1, 0, -1),
                            VoxelIdx(0, 1, 1),
                            VoxelIdx(0, -1, 1), VoxelIdx(0, 1, -1), VoxelIdx(0, -1, -1)};
                    break;
                case Attribute::NeighbourSearchPolicy::NEARBY26:
                    neighbour_voxel_idx_offsets_ = {
                            VoxelIdx(0, 0, 0), VoxelIdx(-1, 0, 0), VoxelIdx(1, 0, 0), VoxelIdx(0, 1, 0),
                            VoxelIdx(0, -1, 0), VoxelIdx(0, 0, -1),
                            VoxelIdx(0, 0, 1), VoxelIdx(1, 1, 0), VoxelIdx(-1, 1, 0), VoxelIdx(1, -1, 0),
                            VoxelIdx(-1, -1, 0),
                            VoxelIdx(1, 0, 1), VoxelIdx(-1, 0, 1), VoxelIdx(1, 0, -1), VoxelIdx(-1, 0, -1),
                            VoxelIdx(0, 1, 1),
                            VoxelIdx(0, -1, 1), VoxelIdx(0, 1, -1), VoxelIdx(0, -1, -1), VoxelIdx(1, 1, 1),
                            VoxelIdx(-1, 1, 1),
                            VoxelIdx(1, -1, 1), VoxelIdx(1, 1, -1), VoxelIdx(-1, -1, 1), VoxelIdx(-1, 1, -1),
                            VoxelIdx(1, -1, -1),
                            VoxelIdx(-1, -1, -1)};
                    break;
                default:
                    std::cout << "Unknown nearby_type!";
            }
        }

        /**
         * add \p point into the object.
         *
         * @param[in] point                     point to be added into object
         */
        void addPoint(const pcl::PointXYZ &point) {
            const VoxelIdx voxel_idx = computeVoxelIdx(point);
            // add point into voxel
            if (voxel_map_.find(voxel_idx) == voxel_map_.end()) {
                const auto new_voxel_iterator = addNewVoxelToListFront(voxel_idx);
                new_voxel_iterator->addPoint(point);
                cullRedundantPoints();
            } else {
                const auto voxel_iterator = moveVoxelToListFront(voxel_idx);
                voxel_iterator->addPoint(point);
            }
            // update occupancy map
            const auto occupied_cell = std::make_pair(voxel_idx.x(), voxel_idx.y());
            occupied_cells_.insert(occupied_cell);
            occupied_range_.addPoint(occupied_cell);
        }

        /**
        * add \p points into the object.
        *
        * @param[in] points                     points to be added into object
        */
        void addPoints(const PointVector &points) {
            std::for_each(
                    std::execution::unseq, points.begin(), points.end(),
                    [this](const auto &point) {
                        addPoint(point);
                    });
        }

        /**
         * add new feature of this object.
         *
         * @param feature                       new feature
         */
        void addFeature(const std::vector<double> &feature) {
            if (features_.size() < MAX_FEATURE_NUM) {
                features_.emplace_back(feature);
            } else {
                std::size_t nearest_feature_idx = -1;
                auto max_similarity = 0;
                for (int feature_idx = 0; feature_idx < features_.size(); ++feature_idx) {
                    const auto feature_similarity = computeFeatureSimilarity(features_.at(feature_idx), feature);
                    if (feature_similarity > max_similarity) {
                        nearest_feature_idx = feature_idx;
                        max_similarity = feature_similarity;
                    }
                }
                features_.at(nearest_feature_idx) = feature;
            }
        }

        /**
         * search for at most \p k nearest neighbour points around \p target_point, and save them into \p closest_points, each
         * distance should not more than \p max_distance.
         *
         * @param[in] target_point              search around this point
         * @param[out] closest_points           save found neighbour points in this vector
         * @param[in] k                         max result size
         * @param[in] max_distance              max distance of each found point
         * @return                              find any point or not
         */
        bool getKNearestNeighbourPoints(
                const pcl::PointXYZ &target_point, PointVector &closest_points, int k = 5,
                double max_distance = 5.0) const {
            // distance info of candidate points
            std::vector<PointDistanceInfo> candidate_distance_infos;
            candidate_distance_infos.reserve(k * neighbour_voxel_idx_offsets_.size());
            // traverse neighbour voxels, search candidate distances
            const auto target_voxel_idx = computeVoxelIdx(target_point);
            for (const VoxelIdx &neighbour_voxel_idx_offset: neighbour_voxel_idx_offsets_) {
                const auto neighbour_voxel_idx = target_voxel_idx + neighbour_voxel_idx_offset;
                if (voxel_map_.find(neighbour_voxel_idx) == voxel_map_.end()) {
                    continue;
                }
                voxel_map_.at(neighbour_voxel_idx)->updateDistanceInfoKNN(candidate_distance_infos, target_point, k,
                                                                          max_distance);
            }
            // cull redundant distance infos
            if (candidate_distance_infos.empty()) {
                return false;
            }
            std::cout << "candidate_distance_infos.size()=" << candidate_distance_infos.size() << std::endl;
            if (candidate_distance_infos.size() > k) {
                std::nth_element(candidate_distance_infos.begin(), candidate_distance_infos.begin() + k - 1,
                                 candidate_distance_infos.end());
                candidate_distance_infos.resize(k);
            }
            std::nth_element(candidate_distance_infos.begin(), candidate_distance_infos.begin(),
                             candidate_distance_infos.end());
            // retrieve points from distance infos
            closest_points.clear();
            for (auto &candidate_distance: candidate_distance_infos) {
                closest_points.emplace_back(candidate_distance.getPoint());
            }
            return !closest_points.empty();
        }

        /**
         * search for closest neighbour point around \p target_point, and save it in \p closest_point
         *
         * @param[in] target_point              search around this point
         * @param[out] closest_point            save found neighbour point in this variable
         * @return                              find successfully or not
         */
        bool getNearestNeighbourPoint(const pcl::PointXYZ &target_point, pcl::PointXYZ &closest_point) const {
            PointVector closest_points;
            std::vector<PointDistanceInfo> candidate_distance_infos;
            // traverse neighbour voxels, search candidate distances
            const auto target_voxel_idx = computeVoxelIdx(target_point);
            for (const auto &neighbour_voxel_idx_offset: neighbour_voxel_idx_offsets_) {
                auto neighbour_voxel_idx = target_voxel_idx + neighbour_voxel_idx_offset;
                if (voxel_map_.find(neighbour_voxel_idx) == voxel_map_.end()) {
                    continue;
                }
                voxel_map_.at(neighbour_voxel_idx)->updateDistanceInfoKNN(
                        candidate_distance_infos, target_point, 1, std::numeric_limits<double>::infinity());
            }
            // return closest point
            if (candidate_distance_infos.empty()) {
                return false;
            }
            const auto min_distance_info_iter = std::min_element(
                    candidate_distance_infos.cbegin(), candidate_distance_infos.cend());
            closest_point = min_distance_info_iter->getPoint();
            return true;
        }

        /**
         * search for closest neighbour points around every point in \p target_points, and save them in \p closest_points with
         * the same order with \p target_points.
         *
         * @param[in] target_points             search around these points
         * @param[out] closest_points           save found neighbour points in this vector
         * @param[out] is_success               save if find successfully for each point
         */
        void getNearestNeighbourPoint(
                const PointVector &target_points, PointVector &closest_points, std::vector<bool> is_success) const {
            std::vector<int> point_idxs(target_points.size());
            for (int i = 0; i < target_points.size(); ++i) {
                point_idxs.at(i) = i;
                is_success.at(i) = false;
            }
            closest_points.resize(target_points.size());

            std::for_each(
                    std::execution::par_unseq, point_idxs.begin(), point_idxs.end(),
                    [&target_points, &closest_points, &is_success, this](size_t point_idx) {
                        pcl::PointXYZ pt;
                        if (getNearestNeighbourPoint(target_points.at(point_idx), pt)) {
                            closest_points.at(point_idx) = pt;
                            is_success.at(point_idx) = true;
                        } else {
                            closest_points.at(point_idx) = pcl::PointXYZ();
                            is_success.at(point_idx) = false;
                        }
                    });
        }

        /**
         * sample point vector from object, extract at most \p num_points_per_voxel points from each voxel. \n
         * if \p num_points_per_voxel equals -1, extract all points from all voxels.
         *
         * @param[in] num_points_per_voxel      sample at most \p num_points_per_voxel from each voxel, -1 means get all points from all voxels.
         * @return                              point vector sampled from this object
         */
        PointVector samplePoints(const int num_points_per_voxel = 10) const {
            PointVector points;
            for (auto &voxel: voxel_list_) {
                const auto size = num_points_per_voxel != -1 ?
                                  std::min<int>(voxel.countPointNum(), num_points_per_voxel) : voxel.countPointNum();
                for (int idx = 0; idx < size; idx++)
                    points.push_back(voxel.getPointByIdx(idx));
            }
            return points;
        }

        /**
         * judge whether \p another_object is the same object with this object by iou and feature similarity.
         *
         * @param[in] another_object            another object
         * @return                              whether \p another_object is the same object with this object
         */
        bool isSameObject(const Object &another_object) const {
            // step1.  check iou
            const auto same_voxel_num = std::count_if(
                    another_object.voxel_list_.cbegin(), another_object.voxel_list_.cend(),
                    [this](auto voxel) { return voxel_map_.find(voxel.getVoxelIdx()) != voxel_map_.cend(); });
            const auto intersection_ratio1 = static_cast<double>(same_voxel_num) / countVoxelNum();
            const auto intersection_ratio2 = static_cast<double>(same_voxel_num) / another_object.countVoxelNum();
            if (intersection_ratio1 < 0.5 && intersection_ratio2 < 0.5) {
                return false;
            }
            return true;
//            if (another_object.features_.empty()) {
//                return true;
//            }
//            // step2. check feature similarity
//            for (const auto &another_feature: another_object.features_) {
//                for (const auto &feature: features_) {
//                    if (computeFeatureSimilarity(feature, another_feature) > 0.7) {
//                        return true;
//                    }
//                }
//            }
//            return false;
        }

        /**
         * merge points from \p another_object into this object.
         *
         * @param another_object                object to be merged into this object
         */
        void mergeObject(const Object &another_object) {
            mergePointsFromAnotherObject(another_object);
            mergeFeaturesFromAnotherObject(another_object);
        }

        /**
         * count number of voxels in the object
         *
         * @return                              number of voxels in the object
         */
        std::size_t countVoxelNum() const {
            return voxel_list_.size();
        }

        /**
         * get object type
         *
         * @return                              object type
         */
        ObjectType getObjectType() const {
            return object_type_;
        }

        /**
        * get class index
        *
        * @return                              class index
        */
        std::uint16_t getClassId() const {
            return toClassId(object_type_);
        }

        /**
         * get class name
         *
         * @return                              class name
         */
        std::string getClassName() const {
            return toClassName(object_type_);
        }

        /**
         * get confidence level
         *
         * @return                              confidence level
         */
        float getConfidence() const {
            return confidence_;
        }

        /**
         * output this object into directory \p object_directory.
         *
         * @param[in] object_directory        output to this directory
         * @return                              whether output successfully
         */
        bool outputToFile(const std::string &object_directory) const {
            namespace fs = std::filesystem;
            // create an empty new directory
            if (fs::exists(object_directory)) {
                fs::remove(object_directory);
            }
            fs::create_directories(object_directory);
            // save point cloud
            const auto point_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            for (const auto point: samplePoints(-1)) {
                point_cloud->emplace_back(point);
            }
            const std::string pcd_file_name = fs::path(object_directory) / "point_cloud.pcd";
            if (pcl::io::save(pcd_file_name, *point_cloud) == -1) {
                return false;
            }
            // save attribute file
            const std::string attribute_file_name = fs::path(object_directory) / "attribute.txt";
            std::ofstream output_stream(attribute_file_name);
            if (!output_stream.is_open()) {
                ROS_ERROR_STREAM("cannot write to file: " << attribute_file_name);
                return false;
            }
            output_stream << toClassId(object_type_) << " " << object_id_ << " " << attribute_.voxel_side_length_
                          << " " << attribute_.capacity_ << std::endl;
            output_stream.close();
            return true;
        }

        /**
         * read an object from directory \p object_directory.
         *
         * @param[in] object_directory        read object from this directory
         * @return                              object read from \p object_directory
         */
        Object &fromDirectory(const std::string &object_directory) {
            namespace fs = std::filesystem;
            // load attribute file
            const std::string attribute_file_name = fs::path(object_directory) / "attribute.txt";
            std::ifstream input_stream(attribute_file_name);
            if (!input_stream.is_open()) {
                ROS_ERROR_STREAM("cannot read from file: " << attribute_file_name);
                return *this;
            }
            uint16_t class_id, object_id;
            float voxel_side_length;
            std::size_t capacity;
            input_stream >> class_id >> object_id >> voxel_side_length >> capacity;
            object_type_ = toObjectType(class_id);
            object_id_ = object_id;
            attribute_ = Attribute(voxel_side_length, capacity);
            confidence_ = CONFIDENCE_FROM_FILE;
            // load points
            const std::string pcd_file_name = fs::path(object_directory) / "point_cloud.pcd";
            const auto point_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            pcl::io::load(pcd_file_name, *point_cloud);
            for (const auto &point: point_cloud->points) {
                addPoint(point);
            }
            return *this;
        }

        /**
         * get occupied cells of this object
         *
         * @return                              occupied cells of this object
         */
        const std::unordered_set<std::pair<int, int>, pair_hash> &getOccupiedCells() const {
            return occupied_cells_;
        }

        /**
         * get occupied range of this object
         *
         * @return                              occupied cells of this object
         */
        CellRange2d getOccupiedRange() const {
            return occupied_range_;
        }

    private:
        /**
         * compute voxel idx of \p point.
         *
         * @param[in] point                     compute voxel of this point
         * @return
         */
        VoxelIdx computeVoxelIdx(const pcl::PointXYZ &point) const {
            return VoxelIdx {
                    roundToNearest(point.x / attribute_.voxel_side_length_),
                    roundToNearest(point.y / attribute_.voxel_side_length_),
                    roundToNearest(point.z / attribute_.voxel_side_length_)
            };
        };

        /**
         * add new voxel into object. store it into the front of \p voxel_list_.
         *
         * @param[in] voxel_idx                 voxel idx of new voxel
         * @return                              iterator pointing to new voxel
         */
        std::list<Voxel>::iterator addNewVoxelToListFront(const VoxelIdx &voxel_idx) {
            const auto new_voxel_center = voxel_idx.cast<float>() * attribute_.voxel_side_length_;
            voxel_list_.push_front({voxel_idx, new_voxel_center, attribute_.voxel_side_length_});
            const auto new_voxel_iterator = voxel_list_.begin();
            voxel_map_.insert({voxel_idx, new_voxel_iterator});
            return new_voxel_iterator;
        }

        /**
         * move voxel with voexl id \p voxel_idx into the front of \p voxel_list_.
         *
         * @param[in] voxel_idx                 voxel idx of moved voxel
         * @return                              iterator pointing to moved voxel
         */
        std::list<Voxel>::iterator moveVoxelToListFront(const VoxelIdx &voxel_idx) {
            const auto voxel_iterator = voxel_map_.find(voxel_idx)->second;
            voxel_list_.splice(voxel_list_.begin(), voxel_list_, voxel_iterator);
            voxel_map_[voxel_idx] = voxel_list_.begin();
            return voxel_iterator;
        }

        /**
         * merge points from \p another_object into this object.
         *
         * @param[in] another_object            object to be merged into this object
         */
        void mergePointsFromAnotherObject(const Object &another_object) {
            for (const auto &another_voxel: another_object.voxel_list_) {
                const auto another_voxel_idx = another_voxel.getVoxelIdx();
                const auto this_voxel_iterator =
                        voxel_map_.find(another_voxel_idx) != voxel_map_.cend() ?
                        moveVoxelToListFront(another_voxel_idx) : addNewVoxelToListFront(another_voxel_idx);
                this_voxel_iterator->mergeVoxel(another_voxel);
            }
        }

        /**
        * merge features from \p another_object into this object.
        *
        * @param[in] another_object            object to be merged into this object
        */
        void mergeFeaturesFromAnotherObject(const Object &another_object) {
            for (const auto &another_feature: another_object.features_) {
                addFeature(another_feature);
            }
        }

        /**
         * remove redundant voxels to make sure voxel num in object no more than \p attribute_.capacity_. redundant
         * voxels are ones stored in the back part of \p voxel_list_.
         */
        void cullRedundantPoints() {
            while (voxel_map_.size() >= attribute_.capacity_) {
                const auto redundant_voxel = voxel_list_.back();
                voxel_map_.erase(redundant_voxel.getVoxelIdx());
                voxel_list_.pop_back();
            }
        }

        /// object type
        ObjectType object_type_;

        /// confidence level(0~1)
        float confidence_;

        /// object index
        uint16_t object_id_;

        /// object attribute
        Attribute attribute_;

        /// voxel lists, keep voxels in LRU order
        std::list<Voxel> voxel_list_;

        /// map from voxel idx into voxel iterator
        std::unordered_map<VoxelIdx, std::list<Voxel>::iterator, voxelIdxHash> voxel_map_;

        /// voxel idx offset of neighbour voxels used for search
        std::vector<VoxelIdx> neighbour_voxel_idx_offsets_;

        /// occupied cells of this object
        std::unordered_set<std::pair<int, int>, pair_hash> occupied_cells_;

        /// occupied range of this object
        CellRange2d occupied_range_;

        /// features of this object
        std::vector<Feature> features_;
    };

} // namespace SemanticMap

#endif  // SEMANTIC_MAP_OBJECT_HPP
