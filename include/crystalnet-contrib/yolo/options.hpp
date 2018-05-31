#pragma once

#include <experimental/filesystem>

struct options_t {
    std::experimental::filesystem::path filename;
    std::experimental::filesystem::path model_dir;
    std::experimental::filesystem::path darknet_path;
    options_t();
};

extern options_t parse_flags(int /* argc */, char *argv[]);
