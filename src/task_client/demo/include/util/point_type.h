#ifndef SEMANTIC_MAP_UTIL_POINT_TYPE_H
#define SEMANTIC_MAP_UTIL_POINT_TYPE_H

#include <Eigen/Core>
#include <pcl/point_cloud.h>

namespace SemanticMap {

    template<typename PointType>
    inline PointType toPclPoint(const Eigen::Vector3d &point_eigen) {
        PointType point_pcl;
        point_pcl.x = point_eigen.x();
        point_pcl.y = point_eigen.y();
        point_pcl.z = point_eigen.z();
        return point_pcl;
    }

    template<typename PointType>
    inline Eigen::Vector3d toEigenPoint(const PointType &point_pcl) {
        return Eigen::Vector3d{point_pcl.x, point_pcl.y, point_pcl.z};
    }

    template<typename PointVector>
    inline typename pcl::PointCloud<typename PointVector::value_type>::Ptr
    toPointCloud(const PointVector &point_vector) {
        using PointType = typename PointVector::value_type;
        auto point_cloud = pcl::make_shared<pcl::PointCloud<PointType>>();

        for (const auto &point: point_vector) {
            point_cloud->push_back(point);
        }
        return point_cloud;
    }

    template<typename PointCloud>
    inline std::vector<typename PointCloud::element_type::PointType> toPointVector(const PointCloud &point_cloud) {
        using PointType = typename PointCloud::element_type::PointType;
        std::vector<PointType> point_vector;

        for (const auto &point: point_cloud->points) {
            point_vector.template emplace_back(point);
        }
        return point_vector;
    }

}

#endif // SEMANTIC_MAP_UTIL_POINT_TYPE_H
