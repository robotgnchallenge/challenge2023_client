#ifndef SEMANTIC_MAP_UTIL_FEATURE_H
#define SEMANTIC_MAP_UTIL_FEATURE_H

#include <Eigen/Core>

namespace SemanticMap {

    using Feature = std::vector<double>;

    ///> length of each feature
    static constexpr int FEATURE_SIZE = 512;

    /**
     * compute cosine similarity between \p feature1 and \p feature2
     *
     * @param[in] feature1
     * @param[in] feature2
     * @return                      cosine similarity between \p feature1 and \p feature2, in [-1, 1]
     */
    double computeFeatureSimilarity(const Feature &feature1, const Feature &feature2) {
        const Eigen::Map<const Eigen::Matrix<double, FEATURE_SIZE, 1>> feature_eigen1(feature1.data(), FEATURE_SIZE);
        const Eigen::Map<const Eigen::Matrix<double, FEATURE_SIZE, 1>> feature_eigen2(feature2.data(), FEATURE_SIZE);
        return feature_eigen1.dot(feature_eigen2) / (feature_eigen1.norm() * feature_eigen2.norm());
    }

}

#endif // SEMANTIC_MAP_UTIL_FEATURE_H
