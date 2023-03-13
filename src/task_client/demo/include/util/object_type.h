#ifndef SEMANTIC_MAP_UTIL_OBJECT_TYPE_H
#define SEMANTIC_MAP_UTIL_OBJECT_TYPE_H

#include <string>
#include <unordered_map>

namespace SemanticMap {

    enum class ObjectType {
        Chair,
        Table,
        Bed,
        PottedTree,
        Cabinet,
        BAR,
        Armchair,
        Sofa,
        LandscapePainting,
        FloorLight,
        BunchOfSunFlowers,
        Crackers,
        Sugar,
        Can,
        Mustard,
        Spam,
        Banana,
        Bowl,
        Mug,
        Drill,
        Scissor,
        Strawberry,
        Apple,
        Lemon,
        Peach,
        Pear,
        Orange,
        Plum,
        Screwdriver,
        Ball,
        Toy,
        Wall,
        BACKGROUND,
        UNKNOWN
    };


    uint16_t toClassId(const ObjectType object_type) {
        switch (object_type) {
            case ObjectType::Chair:
                return 101;
            case ObjectType::Table:
                return 102;
            case ObjectType::Bed:
                return 103;
            case ObjectType::PottedTree:
                return 104;
            case ObjectType::Cabinet:
                return 105;
            case ObjectType::BAR:
                return 106;
            case ObjectType::Armchair:
                return 107;
            case ObjectType::Sofa:
                return 108;
            case ObjectType::LandscapePainting:
                return 109;
            case ObjectType::FloorLight:
                return 110;
            case ObjectType::BunchOfSunFlowers:
                return 111;
            case ObjectType::Crackers:
                return 0;
            case ObjectType::Sugar:
                return 1;
            case ObjectType::Can:
                return 2;
            case ObjectType::Mustard:
                return 3;
            case ObjectType::Spam:
                return 4;
            case ObjectType::Banana:
                return 5;
            case ObjectType::Bowl:
                return 6;
            case ObjectType::Mug:
                return 7;
            case ObjectType::Drill:
                return 8;
            case ObjectType::Scissor:
                return 9;
            case ObjectType::Strawberry:
                return 11;
            case ObjectType::Apple:
                return 12;
            case ObjectType::Lemon:
                return 13;
            case ObjectType::Peach:
                return 14;
            case ObjectType::Pear:
                return 15;
            case ObjectType::Orange:
                return 16;
            case ObjectType::Plum:
                return 17;
            case ObjectType::Screwdriver:
                return 18;
            case ObjectType::Ball:
                return 21;
            case ObjectType::Toy:
                return 25;
            case ObjectType::Wall:
                return 100;
            case ObjectType::BACKGROUND:
                return 200;
            default:
                return -1;
        }
    }


    ObjectType toObjectType(const uint16_t class_id) {
        switch (class_id) {
            case 101:
                return ObjectType::Chair;
            case 102:
                return ObjectType::Table;
            case 103:
                return ObjectType::Bed;
            case 104:
                return ObjectType::PottedTree;
            case 105:
                return ObjectType::Cabinet;
            case 106:
                return ObjectType::BAR;
            case 107:
                return ObjectType::Armchair;
            case 108:
                return ObjectType::Sofa;
            case 109:
                return ObjectType::LandscapePainting;
            case 110:
                return ObjectType::FloorLight;
            case 111:
                return ObjectType::BunchOfSunFlowers;
            case 0:
                return ObjectType::Crackers;
            case 1:
                return ObjectType::Sugar;
            case 2:
                return ObjectType::Can;
            case 3:
                return ObjectType::Mustard;
            case 4:
                return ObjectType::Spam;
            case 5:
                return ObjectType::Banana;
            case 6:
                return ObjectType::Bowl;
            case 7:
                return ObjectType::Mug;
            case 8:
                return ObjectType::Drill;
            case 9:
                return ObjectType::Scissor;
            case 11:
                return ObjectType::Strawberry;
            case 12:
                return ObjectType::Apple;
            case 13:
                return ObjectType::Lemon;
            case 14:
                return ObjectType::Peach;
            case 15:
                return ObjectType::Pear;
            case 16:
                return ObjectType::Orange;
            case 17:
                return ObjectType::Plum;
            case 18:
                return ObjectType::Screwdriver;
            case 21:
                return ObjectType::Ball;
            case 25:
                return ObjectType::Toy;
            case 100:
                return ObjectType::Wall;
            case 200:
                return ObjectType::BACKGROUND;
            default:
                return ObjectType::UNKNOWN;
        }
    }

    std::string toClassName(const ObjectType object_type) {
        static const std::unordered_map<ObjectType, std::string> class_names{
                {ObjectType::Chair,             "Chair"},
                {ObjectType::Table,             "Table"},
                {ObjectType::Bed,               "Bed"},
                {ObjectType::PottedTree,        "PottedTree"},
                {ObjectType::Cabinet,           "Cabinet"},
                {ObjectType::BAR,               "BAR"},
                {ObjectType::Armchair,          "Armchair"},
                {ObjectType::Sofa,              "Sofa"},
                {ObjectType::LandscapePainting, "LandscapePainting"},
                {ObjectType::FloorLight,        "FloorLight"},
                {ObjectType::BunchOfSunFlowers, "BunchOfSunFlowers"},
                {ObjectType::Crackers,          "Crackers"},
                {ObjectType::Sugar,             "Sugar"},
                {ObjectType::Can,               "Can"},
                {ObjectType::Mustard,           "Mustard"},
                {ObjectType::Spam,              "Spam"},
                {ObjectType::Banana,            "Banana"},
                {ObjectType::Bowl,              "Bowl"},
                {ObjectType::Mug,               "Mug"},
                {ObjectType::Drill,             "Drill"},
                {ObjectType::Scissor,           "Scissor"},
                {ObjectType::Strawberry,        "Strawberry"},
                {ObjectType::Apple,             "Apple"},
                {ObjectType::Lemon,             "Lemon"},
                {ObjectType::Peach,             "Peach"},
                {ObjectType::Pear,              "Pear"},
                {ObjectType::Orange,            "Orange"},
                {ObjectType::Plum,              "Plum"},
                {ObjectType::Screwdriver,       "Screwdriver"},
                {ObjectType::Ball,              "Ball"},
                {ObjectType::Toy,               "Toy"},
                {ObjectType::Wall,              "Wall"},
                {ObjectType::BACKGROUND,        "background"},
                {ObjectType::UNKNOWN,           "unknown"},

        };
        return class_names.at(object_type);
    }

    std::string toClassName(const uint16_t class_id) {
        return toClassName(toObjectType(class_id));
    }

    // enum class ObjectType {
    //     BACKGROUND,
    //     CHAIR,
    //     COUCH,
    //     FLOWERPOT,
    //     BED,
    //     TOILET,
    //     TELEVISION_SET,
    //     DESK,
    //     TABLE,
    //     CABINET,
    //     CUP,
    //     BOTTLE,
    //     TOY,
    //     MOUSE,
    //     BOWL,
    //     SINK,
    //     REFRIGERATOR,
    //     UNKNOWN
    // };

    // /**
    //  * leave class id 0 and instance id 0 for background.
    //  */
    // uint16_t toClassId(const ObjectType object_type) {
    //     switch (object_type) {
    //         case ObjectType::BACKGROUND:
    //             return 0;
    //         case ObjectType::CHAIR:
    //             return 1;
    //         case ObjectType::COUCH:
    //             return 2;
    //         case ObjectType::FLOWERPOT:
    //             return 3;
    //         case ObjectType::BED:
    //             return 4;
    //         case ObjectType::TOILET:
    //             return 5;
    //         case ObjectType::TELEVISION_SET:
    //             return 6;
    //         case ObjectType::DESK:
    //             return 7;
    //         case ObjectType::TABLE:
    //             return 8;
    //         case ObjectType::CABINET:
    //             return 9;
    //         case ObjectType::CUP:
    //             return 10;
    //         case ObjectType::BOTTLE:
    //             return 11;
    //         case ObjectType::TOY:
    //             return 12;
    //         case ObjectType::MOUSE:
    //             return 13;
    //         case ObjectType::BOWL:
    //             return 14;
    //         case ObjectType::SINK:
    //             return 15;
    //         case ObjectType::REFRIGERATOR:
    //             return 16;
    //         default:
    //             return -1;
    //     }
    // }

    // ObjectType toObjectType(const uint16_t class_id) {
    //     switch (class_id) {
    //         case 0:
    //             return ObjectType::BACKGROUND;
    //         case 1:
    //             return ObjectType::CHAIR;
    //         case 2:
    //             return ObjectType::COUCH;
    //         case 3:
    //             return ObjectType::FLOWERPOT;
    //         case 4:
    //             return ObjectType::BED;
    //         case 5:
    //             return ObjectType::TOILET;
    //         case 6:
    //             return ObjectType::TELEVISION_SET;
    //         case 7:
    //             return ObjectType::DESK;
    //         case 8:
    //             return ObjectType::TABLE;
    //         case 9:
    //             return ObjectType::CABINET;
    //         case 10:
    //             return ObjectType::CUP;
    //         case 11:
    //             return ObjectType::BOTTLE;
    //         case 12:
    //             return ObjectType::TOY;
    //         case 13:
    //             return ObjectType::MOUSE;
    //         case 14:
    //             return ObjectType::BOWL;
    //         case 15:
    //             return ObjectType::SINK;
    //         case 16:
    //             return ObjectType::REFRIGERATOR;
    //         default:
    //             return ObjectType::UNKNOWN;
    //     }
    // }

    // std::string toClassName(const ObjectType object_type) {
    //     static const std::unordered_map<ObjectType, std::string> class_names{
    //             {ObjectType::BACKGROUND,     "background"},
    //             {ObjectType::CHAIR,          "chair"},
    //             {ObjectType::COUCH,          "couch"},
    //             {ObjectType::FLOWERPOT,      "flowerpot"},
    //             {ObjectType::BED,            "bed"},
    //             {ObjectType::TOILET,         "toilet"},
    //             {ObjectType::TELEVISION_SET, "television_set"},
    //             {ObjectType::DESK,           "desk"},
    //             {ObjectType::TABLE,          "table"},
    //             {ObjectType::CABINET,        "cabinet"},
    //             {ObjectType::CUP,            "cup"},
    //             {ObjectType::BOTTLE,         "bottle"},
    //             {ObjectType::TOY,            "toy"},
    //             {ObjectType::MOUSE,          "mouse"},
    //             {ObjectType::BOWL,           "bowl"},
    //             {ObjectType::SINK,           "sink"},
    //             {ObjectType::REFRIGERATOR,   "refrigerator"},
    //             {ObjectType::UNKNOWN,        "unknown"},
    //     };
    //     return class_names.at(object_type);
    // }



}

#endif // SEMANTIC_MAP_UTIL_OBJECT_TYPE_H
