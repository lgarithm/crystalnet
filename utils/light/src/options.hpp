#pragma once

#include <experimental/filesystem>

struct options_t {
    std::experimental::filesystem::path model_cfg;
    std::experimental::filesystem::path model_dir;
    std::experimental::filesystem::path weights_file;

    options_t();
};

options_t parse_flags(int /* argc */, char *argv[]);
