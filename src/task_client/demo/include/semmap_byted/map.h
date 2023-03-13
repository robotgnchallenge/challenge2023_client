#ifndef SEMANTIC_MAP_MAP_HPP
#define SEMANTIC_MAP_MAP_HPP

#include "object.h"
#include "prior_knowledge.h"

#include <Eigen/Core>
#include <nav_msgs/OccupancyGrid.h>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/make_shared.h>
#include <ros/console.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace SemanticMap {

    /**
     * camera parameters.
     */
    struct Camera {
        int image_width_, image_height_;
        double fx_, fy_, cx_, cy_;
        double k1_, k2_, k3_, p1_, p2_;
        Eigen::Quaterniond rotation_world_camera_;      // transform point from camera coordinate into world coordinate
        Eigen::Vector3d translation_world_camera_;

        Camera(const int image_width, const int image_height,
               const double fx, const double fy, const double cx, const double cy,
               const double k1, const double k2, const double k3, const double p1, const double p2,
               const Eigen::Quaterniond &rotation_world_camera, const Eigen::Vector3d &translation_world_camera)
                : image_width_(image_width), image_height_(image_height),
                  fx_(fx), fy_(fy), cx_(cx), cy_(cy),
                  k1_(k1), k2_(k2), k3_(k3), p1_(p1), p2_(p2),
                  rotation_world_camera_(rotation_world_camera), translation_world_camera_(translation_world_camera) {}

        /**
         * compute pixel coordinate of a 3d point \p point_in_camera.
         *
         * @param point_in_camera               3d point in camera coordinate
         * @return                              pixel coordinate of the point
         */
        std::pair<int, int> computePixelCoordinate(const Eigen::Vector3d &point_in_camera) const {
            const auto x_unified = point_in_camera.x() / point_in_camera.z();
            const auto y_unified = point_in_camera.y() / point_in_camera.z();
            const auto r_square = x_unified * x_unified + y_unified * y_unified;
            const auto x_radial_distortion =
                    2 * p1_ * x_unified * y_unified + p2_ * (r_square + 2 * x_unified * x_unified);
            const auto y_radial_distortion =
                    2 * p2_ * x_unified * y_unified + p1_ * (r_square + 2 * y_unified * y_unified);
            const auto x_tangential_distortion =
                    x_unified * (k1_ * r_square + k2_ * r_square * r_square + k3_ * r_square * r_square * r_square);
            const auto y_tangential_distortion =
                    y_unified * (k1_ * r_square + k2_ * r_square * r_square + k3_ * r_square * r_square * r_square);
            const auto x_distorted = x_unified + x_radial_distortion + x_tangential_distortion;
            const auto y_distorted = y_unified + y_radial_distortion + y_tangential_distortion;
            const auto u = static_cast<int>(fx_ * x_distorted + cx_);
            const auto v = static_cast<int>(fy_ * y_distorted + cy_);
            return std::make_pair(u, v);
        }

        /**
         * compute coordinate of a 3d point from camera coordinate into world coordinate.
         *
         * @param point_in_camera               3d point in camera coordinate
         * @return                              point in world coordinate
         */
        Eigen::Vector3d transformFromCameraToWorld(const Eigen::Vector3d &point_in_camera) const {
            const Eigen::Vector3d point_in_world = rotation_world_camera_ * point_in_camera + translation_world_camera_;
            return point_in_world;
        }
    };

    class Map {
    public:
        static constexpr uint16_t BACKGROUND_INSTANCE_ID{0};
        static constexpr uint16_t BACKGROUND_OBJECT_ID{0};
        static constexpr float PROBABILITY_FOR_UNKNOWN_CELL = 0.1;

        explicit Map(float resolution) : resolution_(resolution) {
            std::cout << "make map with resolution " << resolution << std::endl;
            // make up background object
            object_type_to_object_ids_[ObjectType::BACKGROUND].insert(BACKGROUND_OBJECT_ID);
            object_id_to_object_.emplace(
                    BACKGROUND_INSTANCE_ID, Object(ObjectType::BACKGROUND, BACKGROUND_OBJECT_ID, resolution_, 1.0f));
        }

        /**
         * add \p point_cloud in lidar coordinate into map.
         *
         * @param[in] point_cloud               add these points into map, in **camera** coordinate
         * @param[in] instance_mask             instance_id of each pixel on camera image, type is CV_U16C1
         * @param[in] msg_instance_ids          instance_ids in ROS message
         * @param[in] class_ids                 class index of each object, the same order with \p msg_instance_ids
         * @param[in] features                  feature of each object, length of every feature is 512
         * @param[in] camera                    camera parameter
         */
        void addSemanticInformation(
                const typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr &point_cloud, const cv::Mat &instance_mask,
                const std::vector<uint16_t> &msg_instance_ids, const std::vector<uint16_t> &class_ids,
                const std::vector<Feature> &features, const Camera &camera) {
            makeUpObjectsOfThisFrame(msg_instance_ids, class_ids, features);
            for (const auto point_in_camera: point_cloud->points) {
                addOnePoint(point_in_camera, instance_mask, camera);
            }
            mergeNewObjects(msg_instance_ids);
            cullEmptyObjects();
        }

        /**
         * sample colorful point for \p target_object_type objects in map, distinct color for every object.
         *
         * @param[in] target_object_type        target object type
         * @param[in] num_points_per_voxel      sample at most \p num_points_per_voxel from each voxel
         * @return                              colorful point cloud of the map
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr samplePointCloudForType(
                const ObjectType target_object_type,
                const int num_points_per_voxel = 10) {
            auto point_cloud_for_show = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            for (const auto &[_, object]: object_id_to_object_) {
                const auto object_type = object.getObjectType();
                if (object_type == ObjectType::BACKGROUND || object_type != target_object_type) {
                    continue;
                }
                // generate random color
                const uint8_t color_r = std::rand() % 255;
                const uint8_t color_g = std::rand() % 255;
                const uint8_t color_b = std::rand() % 255;
                // put sample points from each voxel together
                const auto points_from_object = object.samplePoints(num_points_per_voxel);
                for (const auto point: points_from_object) {
                    pcl::PointXYZRGB color_point;
                    color_point.x = point.x;
                    color_point.y = point.y;
                    color_point.z = point.z;
                    color_point.r = color_r;
                    color_point.g = color_g;
                    color_point.b = color_b;
                    point_cloud_for_show->push_back(color_point);
                }
            }
            return point_cloud_for_show;
        }

        /**
         * sample colorful point objects in map, distinct color for each type of object.
         *
         * @param[in] num_points_per_voxel      sample at most \p num_points_per_voxel from each voxel
         * @return                              colorful point cloud of the map
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr samplePointCloudForEachType(
                const int num_points_per_voxel = 10) {
            auto point_cloud_for_show = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            for (const auto &[object_type, object_ids]: object_type_to_object_ids_) {
                if (object_type == ObjectType::BACKGROUND) {
                    continue;
                }
                const uint8_t color_r = std::rand() % 255;
                const uint8_t color_g = std::rand() % 255;
                const uint8_t color_b = std::rand() % 255;
                // put sample points from each voxel together
                for (const auto &object_id: object_ids) {
                    const auto &object = object_id_to_object_.at(object_id);
                    const auto points_from_object = object.samplePoints(num_points_per_voxel);
                    for (const auto point: points_from_object) {
                        pcl::PointXYZRGB color_point;
                        color_point.x = point.x;
                        color_point.y = point.y;
                        color_point.z = point.z;
                        color_point.r = color_r;
                        color_point.g = color_g;
                        color_point.b = color_b;
                        point_cloud_for_show->push_back(color_point);
                    }
                }
            }
            return point_cloud_for_show;
        }

        /**
         * sample colorful point objects in map, distinct color for every object.
         *
         * @param[in] num_points_per_voxel      sample at most \p num_points_per_voxel from each voxel
         * @return                              colorful point cloud of the map
         */
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr samplePointCloudForEachObject(
                const int num_points_per_voxel = 10) {
            auto point_cloud_for_show = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            for (const auto &[_, object]: object_id_to_object_) {
                const auto object_type = object.getObjectType();
                if (object_type == ObjectType::BACKGROUND) {
                    continue;
                }
                // generate random color
                const uint8_t color_r = std::rand() % 255;
                const uint8_t color_g = std::rand() % 255;
                const uint8_t color_b = std::rand() % 255;
                // put sample points from each voxel together
                const auto points_from_object = object.samplePoints(num_points_per_voxel);
                for (const auto point: points_from_object) {
                    pcl::PointXYZRGB color_point;
                    color_point.x = point.x;
                    color_point.y = point.y;
                    color_point.z = point.z;
                    color_point.r = color_r;
                    color_point.g = color_g;
                    color_point.b = color_b;
                    point_cloud_for_show->push_back(color_point);
                }
            }
            return point_cloud_for_show;
        }

        /**
        * get occupancy grid of a given object type \p object_type, computed by the \p object_type and prior knowledge.
        *
        * @param object_type                    query occupancy grid of this grid
        * @return                               occupancy grid in ROS message
        */
        nav_msgs::OccupancyGrid getRosOccupancyGrid(const ObjectType object_type) {
            // compute occupancy range
            CellRange2d grid_range;
            for (const auto &[_, object]: object_id_to_object_) {
                grid_range.mergeRange(object.getOccupiedRange());
            }
            // compute negative probabilities for objects of this class type
            std::unordered_map<std::pair<int, int>, float, pair_hash> negative_probabilities;
            for (const auto object_id: object_type_to_object_ids_[object_type]) {
                const auto &object = object_id_to_object_.at(object_id);
                const auto &occupied_cells = object.getOccupiedCells();
                for (const auto &occupied_cell: occupied_cells) {
                    negative_probabilities[occupied_cell] = 1.0f - object.getConfidence();
                }
            }
            // compute negative probabilities for objects of this correlated type
            const auto relation = PriorKnowledge::getRelation(object_type);
            for (const auto [another_object_type, relation_score]: relation) {
                for (const auto object_id: object_type_to_object_ids_[another_object_type]) {
                    const auto &object = object_id_to_object_.at(object_id);
                    const auto &occupied_cells = object.getOccupiedCells();
                    for (const auto &occupied_cell: occupied_cells) {
                        if (negative_probabilities.find(occupied_cell) == negative_probabilities.end()) {
                            negative_probabilities[occupied_cell] = 1.0f - object.getConfidence() * relation_score;
                        } else {
                            negative_probabilities[occupied_cell] *= (1.0f - object.getConfidence() * relation_score);
                        }
                    }
                }
            }
            // compute negative probabilities for objects of uncorrelated type
            for (const auto &[object_id, objects]: object_id_to_object_) {
                const auto &occupied_cells = objects.getOccupiedCells();
                for (const auto &occupied_cell: occupied_cells) {
                    if (negative_probabilities.find(occupied_cell) == negative_probabilities.end()) {
                        negative_probabilities[occupied_cell] = 1.0f;
                    }
                }
            }
            // get ros occupancy grid map
            nav_msgs::OccupancyGrid ros_map;
            ros_map.info.width = grid_range.get_x_range();
            ros_map.info.height = grid_range.get_y_range();
            ros_map.info.resolution = resolution_;
            ros_map.info.origin.position.x = grid_range.min_x_index_ * resolution_;
            ros_map.info.origin.position.y = grid_range.min_y_index_ * resolution_;
            ros_map.info.origin.position.z = 0.0;
            ros_map.info.origin.orientation.x = 0.0;
            ros_map.info.origin.orientation.y = 0.0;
            ros_map.info.origin.orientation.z = 0.0;
            ros_map.info.origin.orientation.w = 1.0;
            ros_map.data.clear();
            ros_map.data.resize(ros_map.info.width * ros_map.info.height, PROBABILITY_FOR_UNKNOWN_CELL * 100.0f);
            for (const auto [cell_index, negative_probability]: negative_probabilities) {
                const auto x_index_in_data = cell_index.first - grid_range.min_x_index_;
                const auto y_index_in_data = cell_index.second - grid_range.min_y_index_;
                const auto grid_cell_data = static_cast<uint8_t>((1 - negative_probability) * 100.0f);
                ros_map.data.at(ros_map.info.width * y_index_in_data + x_index_in_data) = grid_cell_data;
            }
            return ros_map;
        }

        /**
         * print summary of the map
         */
        void printSummary() {
            std::cout << "map summary:" << std::endl;
            for (const auto &[object_id, object]: object_id_to_object_) {
                std::cout << "object_id=" << std::setw(3) << object_id << ", object_type=" << object.getClassName()
                          << ", voxels_num=" << object.countVoxelNum() << std::endl;
            }
            std::cout << "--------" << std::endl;
        }

        /**
         * save the map into \p map_directory. \n
         * the root directory save for entire map, each subdirectory save for a class, and each subsubdirectory save for
         * an object.
         *
         * @param[in] map_directory             map are saved in this directory
         * @return                              whether output successfully
         */
        bool outputToDirectory(const std::string &map_directory) const {
            ROS_INFO_STREAM("save map into " << map_directory);
            namespace fs = std::filesystem;
            // create an empty new directory
            if (fs::exists(map_directory)) {
                fs::remove_all(map_directory);
            }
            fs::create_directories(map_directory);
            // save attribute file
            const std::string attribute_file_name = fs::path(map_directory) / "attribute.txt";
            std::ofstream output_stream(attribute_file_name);
            if (!output_stream.is_open()) {
                ROS_ERROR_STREAM("cannot write to file: " << attribute_file_name);
                return false;
            }
            output_stream << resolution_ << std::endl;
            output_stream.close();
            // save every object to file
            for (const auto &[object_type, object_ids]: object_type_to_object_ids_) {
                const auto class_directory = fs::path(map_directory) / std::to_string(toClassId(object_type));
                fs::create_directories(class_directory);
                for (const auto &object_id: object_ids) {
                    const auto object_directory = class_directory / std::to_string(object_id);
                    const auto &object = object_id_to_object_.at(object_id);
                    // don't save empty objects
                    if (object.countVoxelNum() == 0) {
                        ROS_WARN("have empty object, what the fuck?");
                        continue;
                    }
                    if (!object.outputToFile(object_directory)) {
                        return false;
                    }
                }
            }
            ROS_INFO_STREAM("save map successfully");
            return true;
        }

        /**
         * read map from directory \p map_directory.
         *
         * @param[in] map_directory             read map from this directory
         * @return                              map read from \p map_directory
         */
        Map &fromDirectory(const std::string &map_directory) {
            namespace fs = std::filesystem;
            // load attribute file
            const std::string attribute_file_name = fs::path(map_directory) / "attribute.txt";
            std::ifstream input_stream(attribute_file_name);
            if (!input_stream.is_open()) {
                ROS_ERROR_STREAM("cannot read from file: " << attribute_file_name);
                return *this;
            }
            float resolution;
            input_stream >> resolution;
            if (resolution != resolution_) {
                ROS_ERROR(
                        "map resolution not match: resolution in launch file is %f, but resolution in map directory is %f",
                        resolution_, resolution);
                return *this;
            }
            // load objects
            using dir_iter = fs::directory_iterator;
            for (auto class_entry = dir_iter(map_directory); class_entry != dir_iter(); ++class_entry) {
                const auto class_directory = class_entry->path();
                if (!class_entry->is_directory()) {
                    continue;
                }
                const auto object_type = toObjectType(static_cast<uint16_t>(std::stoi(class_directory.stem())));
                for (auto object_entry = dir_iter(class_directory); object_entry != dir_iter(); ++object_entry) {
                    const auto object_directory = object_entry->path();
                    const auto object_id = static_cast<uint16_t>(std::stoi(object_directory.filename()));
                    object_id_to_object_[object_id].fromDirectory(object_directory);
                    object_type_to_object_ids_[object_type].insert(object_id);
                }
            }
            return *this;
        }

        void clear()
        {
            object_id_to_object_.clear();
            object_type_to_object_ids_.clear();
            instance_id_to_object_id_.clear();
            object_type_to_object_ids_[ObjectType::BACKGROUND].insert(BACKGROUND_OBJECT_ID);
            object_id_to_object_.emplace(
                    BACKGROUND_INSTANCE_ID, Object(ObjectType::BACKGROUND, BACKGROUND_OBJECT_ID, resolution_, 1.0f));
            
        }


    private:
        /**
         * make up empty objects of this frame.
         *
         * @param[in] msg_instance_ids          instance_ids in ROS message
         * @param[in] class_ids                 class_id of each object, the same order with \p msg_instance_ids
         * @param[in] features                  feature of each object, length of every feature is 512
         * @return
         */
        void makeUpObjectsOfThisFrame(
                const std::vector<uint16_t> &msg_instance_ids, const std::vector<uint16_t> &class_ids,
                const std::vector<Feature> &features) {
            assert(msg_instance_ids.size() == class_ids.size());

            instance_id_to_object_id_.clear();
            instance_id_to_object_id_.emplace(BACKGROUND_INSTANCE_ID, BACKGROUND_OBJECT_ID);

            for (int idx = 0; idx < msg_instance_ids.size(); ++idx) {
                const auto msg_instance_id = msg_instance_ids.at(idx);
                if (msg_instance_id == BACKGROUND_INSTANCE_ID) {
                    continue;
                }
                const auto object_id = next_object_id_++;
                const auto object_type = toObjectType(class_ids.at(idx));
                instance_id_to_object_id_.emplace(msg_instance_id, object_id);
                object_id_to_object_.emplace(
                        object_id, Object(object_type, object_id, resolution_, Object::CONFIDENCE_FROM_POINT_CLOUD));
                object_type_to_object_ids_[object_type].insert(object_id);
                if (!features.empty()) {
                    object_id_to_object_.at(object_id).addFeature(features.at(idx));
                }
            }
        }

        /**
         * add \p point_in_lidar into map.
         *
         * @param[in] point_in_lidar            point to be added, in lidar coordinate
         * @param[in] instance_mask             instance_id of each pixel on camera image
         * @param[in] camera                    camera parameter
         */
        void addOnePoint(const pcl::PointXYZ &point_in_lidar, const cv::Mat &instance_mask, const Camera &camera) {
            // find projection pixel coordinate
            const Eigen::Vector3d point_in_camera_eigen = toEigenPoint(point_in_lidar);
            const auto [u, v] = camera.computePixelCoordinate(point_in_camera_eigen);
            if (u < 0 || u >= camera.image_width_ || v < 0 || v >= camera.image_height_) {
                return;
            }
            // get instance_id and object_id
            const auto msg_instance_id = instance_mask.at<uint16_t>(v, u);
            const auto object_id = instance_id_to_object_id_.at(msg_instance_id);
            // add point_in_world into object
            Eigen::Vector3d point_in_world = camera.transformFromCameraToWorld(point_in_camera_eigen);
            object_id_to_object_.at(object_id).addPoint(toPclPoint<pcl::PointXYZ>(point_in_world));
        }

        /**
        * remove empty objects.
        */
        void cullEmptyObjects() {
            // find empty object_ids
            std::unordered_set<uint16_t> empty_object_ids;
            for (const auto &[object_id, object]: object_id_to_object_) {
                if (object.countVoxelNum() == 0) {
                    empty_object_ids.insert(object_id);
                }
            }
            // remove empty objects
            for (const auto &object_id: empty_object_ids) {
                const auto object_type = object_id_to_object_.at(object_id).getObjectType();
                object_id_to_object_.erase(object_id);
                object_type_to_object_ids_.at(object_type).erase(object_id);
            }
        }

        /**
         * merge new objects into existing objects.
         *
         * @param[in] msg_instance_ids          instance_ids in ROS message of new objects
         */
        void mergeNewObjects(const std::vector<uint16_t> &msg_instance_ids) {
            std::unordered_set<uint16_t> deleted_object_ids;
            for (const auto &new_msg_instance_id: msg_instance_ids) {
                if (new_msg_instance_id == BACKGROUND_INSTANCE_ID) {
                    continue;
                }
                const auto new_object_id = instance_id_to_object_id_.at(new_msg_instance_id);
                const auto &new_object = object_id_to_object_.at(new_object_id);
                const auto object_type = new_object.getObjectType();
                for (const auto &object_id: object_type_to_object_ids_.at(object_type)) {
                    if (object_id == new_object_id || deleted_object_ids.find(object_id) != deleted_object_ids.end()) {
                        continue;
                    }
                    auto &object = object_id_to_object_.at(object_id);
                    if (object.isSameObject(new_object)) {
                        object.mergeObject(new_object);
                        deleted_object_ids.insert(new_object_id);
                    }
                }
            }
            for (const auto &deleted_object_id: deleted_object_ids) {
                const auto object_type = object_id_to_object_.at(deleted_object_id).getObjectType();
                object_id_to_object_.erase(deleted_object_id);
                object_type_to_object_ids_.at(object_type).erase(deleted_object_id);
            }
        }

        /// resolution of map voxels
        float resolution_;

        /// from object_id in map into objects
        std::unordered_map<uint16_t, Object> object_id_to_object_;

        /// map from object type to object_ids in map
        std::unordered_map<ObjectType, std::unordered_set<uint16_t>> object_type_to_object_ids_;

        /// from instance_id in ROS message into object_id in map
        std::unordered_map<uint16_t, uint16_t> instance_id_to_object_id_;

        /// object_id of new object, start from 1, leave 0 for background
        uint16_t next_object_id_{1};
    };

} // namespace SemanticMap

#endif // SEMANTIC_MAP_MAP_HPP
