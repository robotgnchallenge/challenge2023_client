#ifndef SEMANTIC_MAP_PRIOR_KNOWLEDGE_H
#define SEMANTIC_MAP_PRIOR_KNOWLEDGE_H

#include "util.h"

#include <unordered_map>

namespace SemanticMap {

    class PriorKnowledge {
    public:
        static std::unordered_map<ObjectType, float> getRelation(const ObjectType object_name) {
            std::unordered_map<ObjectType, float> relation_probability;
            relation_probability[ObjectType::Table] = 0.35;
            // relation_probability[ObjectType::DESK] = 0.35;
            // relation_probability[ObjectType::CABINET] = 0.3;
            return relation_probability;
        }

        // load prior knowledge from file
        void loadPriorKnowledge(const std::string &config_file_name) {

        }
    };
}

#endif //SEMANTIC_MAP_PRIOR_KNOWLEDGE_H
