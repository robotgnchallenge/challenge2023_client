#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ros/console.h>
#include <ros/duration.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/publisher.h>
#include <semmap_byted/map.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/UInt16.h>
#include <task_client/set_int.h>
#include <task_client/set_intRequest.h>
#include <task_client/set_intResponse.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Core>
#include <boost/shared_ptr.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "task_client/InstanceInfo.h"

#include "codecvt"
#include "locale"
#include "util.h"

namespace py = pybind11;

py::scoped_interpreter python;

namespace {
enum VisualizeMode { kTargetMode = 0, kObjectType = 1, kObjectId = 2 };
}

class PointProjector {
 public:
  explicit PointProjector(const std::string& camera_intrinsic_file_name)
      : camera_intrinsic_(CameraIntrinsic::fromFile(camera_intrinsic_file_name)) {
    std::string home_dir = getenv("HOME");
    this->sys_ = py::module::import("sys");
    this->sys_.attr("path").attr("append")(home_dir + "/client_ws/src/task_client/demo/src");
    this->task_client_ = py::module::import("task_client_socket").attr("task_client_socket")();
  }

  void RequestSensorInfo() {
    int i = 0;
    while (i <= 5) {
      try {
        ++i;
        py::object sensor_info_object =
            this->task_client_.attr("request_info")("REQUEST_SENSOR_INFO --depth --seg --head");
        py::dict sensor_info_dict = sensor_info_object.cast<py::dict>();
        this->depth.empty();
        this->instance.empty();
        for (auto item : sensor_info_dict) {
          auto key = item.first.cast<std::string>();
          if (key == "depth_head") {
            auto depth_dict = item.second.cast<py::dict>();
            for (auto sec : depth_dict) {
              auto info = sec.first.cast<std::string>();
              if (info == "data") {
                auto depth_buf = sec.second.cast<py::array_t<float>>().request();
                cv::Mat depth_mat(depth_buf.shape[0], depth_buf.shape[1], CV_32FC1, (float*)depth_buf.ptr);
                this->depth = depth_mat.clone();
              }
              if (info == "time") {
                std::string time = sec.second.cast<std::string>();
                std::stringstream ss(time);
                ::uint64_t nsec;
                if (!(ss >> nsec)) {
                  throw std::runtime_error("failed to convert to uint32");
                }

                this->depth_time.fromNSec(nsec);
              }
            }
          }
          if (key == "seg_head") {
            auto instance_dict = item.second.cast<py::dict>();
            for (auto sec : instance_dict) {
              auto info = sec.first.cast<std::string>();
              if (info == "data") {
                auto instance_buf = sec.second.cast<py::array_t<float>>().request();
                cv::Mat instance_mat(instance_buf.shape[0], instance_buf.shape[1], CV_32FC1, (float*)instance_buf.ptr);
                this->instance = instance_mat.clone();
              }
              if (info == "time") {
                std::string time = sec.second.cast<std::string>();
                std::stringstream ss(time);
                ::uint64_t nsec;
                if (!(ss >> nsec)) {
                  throw std::runtime_error("failed to convert to uint32");
                }

                this->instance_time.fromNSec(nsec);
              }
            }
          }
        }

        ros::Rate(5.0).sleep();
        break;

      } catch (std::exception& e) {
        if (i == 5) {
          this->task_client_.attr("reconnect")();
          i = 0;
        }
        continue;
      }
    }
  }

  std::pair<Eigen::Quaterniond, Eigen::Vector3d> LookupTransformSocket(const std::string& source_frame,
                                                                       const std::string& target_frame,
                                                                       ros::Time timestamp = ros::Time::now()) {
    auto sec = timestamp.sec;
    auto nsec = timestamp.nsec;

    std::stringstream ss_sec;
    std::stringstream ss_nsec;

    ss_sec << sec;
    ss_nsec << nsec;
    std::string string_sec = ss_sec.str();
    std::string string_nsec = ss_nsec.str();

    std::string string_timestamp = "[" + string_sec + "," + string_nsec + "]";
    std::string request_msg = std::string("[TF]") + std::string("LOOKUP_TRANSFORM ") + std::string("--source ") +
                              source_frame + " " + std::string("--target ") + target_frame + " " +
                              std::string("--time ") + string_timestamp;

    while (true) {
      try {
        py::object tf_transfomer_object = this->task_client_.attr("request_info")(request_msg);
        auto tf_transfomer_array = tf_transfomer_object.cast<py::array_t<float>>();
        std::vector<float> trans_rot;
        auto buf = tf_transfomer_array.request();
        for (int i = 0; i < buf.shape[0]; i++) {
          trans_rot.emplace_back(*((float*)buf.ptr + i));
        }

        Eigen::Quaterniond rotation(trans_rot[6], trans_rot[3], trans_rot[4], trans_rot[5]);
        Eigen::Vector3d translation(trans_rot[0], trans_rot[1], trans_rot[2]);

        return {rotation, translation};
        break;
      } catch (std::exception& e) {
        this->task_client_.attr("reconnect")();
        continue;
      }
    }
    return {{}, {}};
  }

  void SendProbabilityMap() {
    int width = this->instance.cols;
    int height = this->instance.rows;
    std::vector<uint16_t> instance_ids;
    std::vector<uint16_t> class_ids;
    std::set<int> class_set;
    const std::vector<SemanticMap::Feature> fetures;
    std::vector<std::vector<int>> mask(height, std::vector<int>(width, 0));
    cv::Mat mask_info_mat = cv::Mat::zeros(height, width, CV_16UC1);

    /** Find all class instance in segmentation map*/
    int i = 1;
    for (int x = 0; x < this->instance.cols; ++x) {
      for (int y = 0; y < this->instance.rows; ++y) {
        /** Check depth validity for realsense*/
        int class_id = (int)cvRound(this->instance.at<float>(y, x));
        if (class_id == -1) continue;

        if (class_set.insert(class_id).second) {
          class_ids.emplace_back(class_id);
          instance_ids.emplace_back(i);
          i++;
        }
      }
    }

    /** Generate mask info (position infomation) for all class instance*/
    for (int x = 0; x < this->instance.cols; ++x) {
      for (int y = 0; y < this->instance.rows; ++y) {
        int class_id = (int)cvRound(this->instance.at<float>(y, x));
        if (class_id == -1) continue;
        auto a = instance_ids[find(class_ids.begin(), class_ids.end(), class_id) - class_ids.begin()];
        mask_info_mat.at<ushort>(y, x) =
            instance_ids[find(class_ids.begin(), class_ids.end(), class_id) - class_ids.begin()];
      }
    }

    /** Dilate mask and extract object point cloud*/
    cv::Mat element;
    cv::Mat Dilate;
    cv::Mat Erode;
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    erode(mask_info_mat, mask_info_mat, element);
    SemanticMap::TimeProfiler time_counter;
    auto time_start = std::chrono::system_clock::now();
    this->depth.convertTo(this->depth, CV_32FC1);
    cv::Mat instance_mask = extractInstanceMask(mask_info_mat, {this->depth.cols, this->depth.rows});
    const double depth_threshold = 4;
    const int sample_step = ros::param::param<int>("sample_step", 2);

    const auto depth_point_cloud =
        extractPointCloudFromDepthImage(instance_mask, this->depth, instance_ids, sample_step, depth_threshold);

    visualizePointCloud(depth_point_cloud, realsense_point_cloud_publisher_, this->depth_time, DEPTH_CAMERA_FRAME_ID);
    time_counter.recordTimePoint("extract point cloud from ROS message");

    /** Lookup transform*/
    const auto [rotation_world_camera, translation_world_camera] =
        LookupTransformSocket(WORLD_FRAME_ID, DEPTH_CAMERA_FRAME_ID, this->instance_time);

    time_counter.recordTimePoint("look up for transform");
    /** Update semantic map*/
    const auto camera = SemanticMap::Camera(
        camera_intrinsic_.image_width, camera_intrinsic_.image_height, camera_intrinsic_.fx, camera_intrinsic_.fy,
        camera_intrinsic_.cx, camera_intrinsic_.cy, camera_intrinsic_.k1, camera_intrinsic_.k2, camera_intrinsic_.k3,
        camera_intrinsic_.p1, camera_intrinsic_.p2, rotation_world_camera, translation_world_camera);
    map_.addSemanticInformation(depth_point_cloud, instance_mask, instance_ids, class_ids, fetures, camera);
    time_counter.recordTimePoint("add point cloud into instances");
    int target_class_id = this->target_id_;
    auto target_object_type = SemanticMap::toObjectType(target_class_id);

    const auto VisualizeMode = ros::param::param<int>("semantic_point_cloud_visualize_mode", 0);
    auto semantic_point_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    switch (VisualizeMode) {
      case kTargetMode:
        semantic_point_cloud = map_.samplePointCloudForType(target_object_type);
        break;
      case kObjectType:
        semantic_point_cloud = map_.samplePointCloudForEachType(10);
        break;
      case kObjectId:
        semantic_point_cloud = map_.samplePointCloudForEachObject(10);
        break;
      default:
        ROS_ERROR("not a valid value of ros param semantic_point_cloud_visualize_mode, check launch file.");
    }

    visualizePointCloud(semantic_point_cloud, semantic_point_cloud_publisher_, this->depth_time);
    time_counter.recordTimePoint("sample and broadcast colorful semantic point cloud");

    /** Show probability map*/
    auto occupancygrid = map_.getRosOccupancyGrid(target_object_type);
    occupancygrid.header.frame_id = "map";
    occupancygrid.header.stamp = this->depth_time;

    probability_map_publisher_.publish(occupancygrid);

    time_counter.recordTimePoint("construct and broadcast probability map");
  }

  bool ChangeObjidCb(task_client::set_int::Request& req, task_client::set_int::Response& res) {
    map_.clear();
    this->target_id_ = req.value;
    std::cout << "[CLIENT_SEGMAP]Clear privious map, Now target is change to " << this->target_id_ << std::endl;

    int target_class_id = this->target_id_;

    auto target_object_type = SemanticMap::toObjectType(target_class_id);

    auto occupancygrid = map_.getRosOccupancyGrid(target_object_type);
    occupancygrid.header.frame_id = "map";
    occupancygrid.header.stamp = this->depth_time;

    probability_map_publisher_.publish(occupancygrid);

    res.success = true;
    return true;
  }

  ros::Publisher realsense_point_cloud_publisher_;

  ros::Publisher semantic_point_cloud_publisher_;

  ros::Publisher probability_map_publisher_;

  ros::Subscriber target_id_sub_;
  uint16_t target_id_{7};

  py::module sys_;
  py::object task_client_;

 private:
  /**
   * Extract features from instance_info msg
   *
   * @param instance_info_msg                     message including instance info
   * @return                                      features from \p instance_info_msg
   */
  std::vector<SemanticMap::Feature> extractFeaturesFromMsg(
      const task_client::InstanceInfo::ConstPtr& instance_info_msg) {
    std::vector<SemanticMap::Feature> features;
    for (const auto& feature : instance_info_msg->features) {
      features.emplace_back(feature.data);
    }
    return features;
  }

  /**
   * Broadcast the transform for point from \p source_frame to \p target_frame, at time \p timestamp.
   *
   * @param[in] target_frame                      transform point from this frame
   * @param[in] source_frame                      transform point into this frame
   * @param[in] rotation                          rotation of the transform
   * @param[in] translation                       translation of the transform
   * @param[in] timestamp                         timestamp of the transform
   */
  void broadcastTransform(const std::string& target_frame, const std::string& source_frame,
                          const Eigen::Quaterniond& rotation, const Eigen::Vector3d& translation,
                          const ros::Time& timestamp) {
    tf::StampedTransform transform;
    transform.frame_id_ = target_frame;
    transform.child_frame_id_ = source_frame;
    transform.setRotation(tf::Quaternion(rotation.x(), rotation.y(), rotation.z(), rotation.w()));
    transform.setOrigin(tf::Vector3(translation.x(), translation.y(), translation.z()));
    transform.stamp_ = timestamp;
    transform_broadcaster_.sendTransform(transform);
  }

  /**
   * Compute valid depth threshold for every instance. used for filtering out depth-inconsistent points when extract
   * point cloud from depth image.
   *
   * @param[in] instance_mask                 instance mask
   * @param[in] depth_image                   depth image
   * @param[in] instance_ids                  instance indices in the mask
   * @param[in] step                          compute every \p step pixels
   * @return                                  instance id to its min depth threshold and max depth threshold in
   * millimeter
   */
  static std::unordered_map<uint16_t, std::pair<float, float>> computeInstanceDepthThreshold(
      const cv::Mat& instance_mask, const cv::Mat& depth_image, const std::vector<uint16_t>& instance_ids,
      const int step) {
    std::unordered_map<uint16_t, std::vector<float>> instance_to_depths;
    static constexpr uint16_t BACKGROUND_INSTANCE_ID = 0;
    for (int y = 0; y < instance_mask.cols; y += step) {
      for (int x = 0; x < instance_mask.rows; x += step) {
        const int instance_id = instance_mask.at<uint16_t>(x, y);
        if (instance_id == BACKGROUND_INSTANCE_ID) {
          continue;
        }
        const auto depth = depth_image.at<float>(x, y);
        instance_to_depths[instance_id].emplace_back(depth);
      }
    }
    std::unordered_map<uint16_t, std::pair<float, float>> depth_thresholds;
    for (const auto& instance_id : instance_ids) {
      depth_thresholds.emplace(instance_id,
                               std::make_pair(std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
    }
    for (auto& [instance_id, depths] : instance_to_depths) {
      const auto pixels_num = depths.size();
      std::nth_element(depths.begin(), depths.begin() + pixels_num / 5, depths.end());
      const auto min_depth = *(depths.cbegin() + pixels_num / 5);
      std::nth_element(depths.begin(), depths.begin() + pixels_num / 5, depths.end(), std::greater<>());
      const auto max_depth = *(depths.cbegin() + pixels_num / 5);
      depth_thresholds.at(instance_id) = {min_depth, max_depth};
    }
    return depth_thresholds;
  }

  /**
   * Get instance mask from a ROS \p instance_info message. do following things for a ROS \p instance_info message: \n
   * 1. get original instance mask. \n
   * 2. if have a valid \p target_size parameter, scale the instance mask into size \p target_size. \n
   *
   * @param instance_info_msg                     input ROS instance_info message
   * @param target_size                           target size of scaling, {0, 0} means no scaling
   * @return
   */
  static cv::Mat extractInstanceMask(const cv::Mat& instance_mask, const cv::Size& target_size = {0, 0}) {
    /** Scale the mask*/
    cv::Mat scaled_instance_mask;
    if (target_size.width == 0) {
      scaled_instance_mask = instance_mask;
    } else {
      cv::resize(instance_mask, scaled_instance_mask, target_size);
    }
    return scaled_instance_mask;
  }

  /**
   * Extract point cloud from \p depth_image from realsense camera. pixels with depth larger than /p
   * depth_threshold_in_millimeter are dropped. we only process pixels every \p step pixels.
   *
   * @param[in] instance_mask                 instance mask
   * @param[in] depth_image                   depth image from realsense camera
   * @param[in] instance_ids                  instance indices in the mask
   * @param[in] step                          compute every \p step pixels
   * @param[in] depth_threshold               pixels with depth larger than this value are dropped, same unit with depth
   * image
   * @return                                  point cloud extracted from \p depth_image
   */
  pcl::PointCloud<pcl::PointXYZ>::Ptr extractPointCloudFromDepthImage(const cv::Mat& instance_mask,
                                                                      const cv::Mat& depth_image,
                                                                      const std::vector<uint16_t>& instance_ids,
                                                                      const int step,
                                                                      const double depth_threshold) const {
    /** Compute depth threshold for each instance*/
    const auto instance_depth_thresholds =
        computeInstanceDepthThreshold(instance_mask, depth_image, instance_ids, step);
    /** Reproject pixels into 3d points*/
    const bool image_depth_in_meter = ros::param::param<bool>("is_virtual_scene", true);
    auto point_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for (int x = 0; x < depth_image.cols; x += step) {
      for (int y = 0; y < depth_image.rows; y += step) {
        /** Check depth validity for realsense*/
        const auto depth_from_image = depth_image.at<float>(y, x);
        if (depth_from_image > depth_threshold || depth_from_image <= 0.6) {
          continue;
        }
        /** Check depth_from_image validity for instance*/
        static constexpr uint16_t BACKGROUND_INSTANCE_ID = 0;
        const int instance_id = instance_mask.at<uint16_t>(y, x);
        if (instance_id != BACKGROUND_INSTANCE_ID) {
          const auto [min_depth_in_milli, max_depth_in_milli] = instance_depth_thresholds.at(instance_id);
          if (depth_from_image < min_depth_in_milli || depth_from_image > max_depth_in_milli) {
            continue;
          }
        }
        /** Reproject 3d point*/
        pcl::PointXYZ point_in_camera;
        const float depth_in_meter = image_depth_in_meter ? depth_from_image : depth_from_image / 1000.0f;
        point_in_camera.z = depth_in_meter;
        point_in_camera.x = (x - camera_intrinsic_.cx) * depth_in_meter / camera_intrinsic_.fx;
        point_in_camera.y = (y - camera_intrinsic_.cy) * depth_in_meter / camera_intrinsic_.fy;
        point_cloud->push_back(point_in_camera);
      }
    }
    return point_cloud;
  }

  /**
   * Show point cloud by rviz.
   */
  template <typename PointCloudPtr>
  void visualizePointCloud(const PointCloudPtr& point_cloud, const ros::Publisher& publisher,
                           const ros::Time& timestamp, const std::string& frame_id = "map") {
    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(*point_cloud, point_cloud_msg);
    point_cloud_msg.header.frame_id = frame_id;
    point_cloud_msg.header.stamp = timestamp;
    publisher.publish(point_cloud_msg);
  }

  struct CameraIntrinsic {
    static CameraIntrinsic fromFile(const std::string& camera_intrinsic_file_name) {
      /** Open config file*/
      cv::FileStorage camera_intrinsic_file(camera_intrinsic_file_name, cv::FileStorage::READ);
      if (!camera_intrinsic_file.isOpened()) {
        ROS_ERROR("config file [%s] doesn't exists.", camera_intrinsic_file_name.c_str());
      }
      /** Read from config file*/
      CameraIntrinsic camera_intrinsic;
      camera_intrinsic_file["fx"] >> camera_intrinsic.fx;
      camera_intrinsic_file["fy"] >> camera_intrinsic.fy;
      camera_intrinsic_file["cx"] >> camera_intrinsic.cx;
      camera_intrinsic_file["cy"] >> camera_intrinsic.cy;
      camera_intrinsic_file["k1"] >> camera_intrinsic.k1;
      camera_intrinsic_file["k2"] >> camera_intrinsic.k2;
      camera_intrinsic_file["p1"] >> camera_intrinsic.p1;
      camera_intrinsic_file["p2"] >> camera_intrinsic.p2;
      camera_intrinsic_file["k3"] >> camera_intrinsic.k3;
      camera_intrinsic_file["image_width"] >> camera_intrinsic.image_width;
      camera_intrinsic_file["image_height"] >> camera_intrinsic.image_height;
      return camera_intrinsic;
    }

    int image_width, image_height;
    double k1, k2, k3, p1, p2;
    double fx, fy, cx, cy;
  };

  cv::Mat depth;
  ros::Time depth_time;
  cv::Mat instance;
  ros::Time instance_time;

  SemanticMap::Map map_{ros::param::param<float>("map_resolution", 0.05)};

  CameraIntrinsic camera_intrinsic_;

  const std::string DEPTH_CAMERA_FRAME_ID{
      ros::param::param<std::string>("depth_camera_frame_id", "/MirKinova/kinect_range")};

  const std::string WORLD_FRAME_ID{ros::param::param<std::string>("world_frame_id", "/map")};

  const std::string ROBOT_FRAME_ID{ros::param::param<std::string>("robot_frame_id", "/base_link")};

  tf::TransformListener transform_listener_;

  tf::TransformBroadcaster transform_broadcaster_;
};

int main(int argc, char** argv) {
  /** Init node*/
  ros::init(argc, argv, "construct_semmap_node");
  ros::NodeHandle nh;

  /** Subscribe input topics*/
  const auto INSTANCE_INFO_TOPIC = ros::param::param<std::string>("instance_info_topic", "/instance_info");
  const auto POINT_CLOUD_TOPIC = ros::param::param<std::string>("point_cloud_topic", "/cloud_effected");
  const auto DEPTH_IMAGE_TOPIC =
      ros::param::param<std::string>("depth_image_topic", "/camera/aligned_depth_to_color/image_raw");
  const auto PROBABILITY_MAP_TOPIC = ros::param::param<std::string>("probability_map_topic", "/probability_map");

  /** Create map constructor*/
  std::string home_dir = getenv("HOME");
  const auto camera_intrinsic_file_name = ros::param::param<std::string>(
      "camera_intrinsic_file_name", home_dir + "/client_ws/src/task_client/config/intrinsic_virtual.yaml");
  PointProjector point_projector(camera_intrinsic_file_name);
  point_projector.probability_map_publisher_ = nh.advertise<nav_msgs::OccupancyGrid>(PROBABILITY_MAP_TOPIC, 10);
  point_projector.semantic_point_cloud_publisher_ = nh.advertise<sensor_msgs::PointCloud2>("semantic_point_cloud", 10);
  ros::ServiceServer service = nh.advertiseService("change_id", &PointProjector::ChangeObjidCb, &point_projector);
  point_projector.realsense_point_cloud_publisher_ =
      nh.advertise<sensor_msgs::PointCloud2>("realsense_point_cloud", 10);

  bool is_get_semantic_map;

  while (ros::ok()) {
    ros::param::param<bool>("/is_get_semantic_map", is_get_semantic_map, false);
    if (is_get_semantic_map) {
      point_projector.RequestSensorInfo();
      point_projector.SendProbabilityMap();
    }

    ros::spinOnce();
  }

  std::cout << "[CLIENT_SEGMAP]Construct semantics map sucessfully." << std::endl;

  return 0;
}
