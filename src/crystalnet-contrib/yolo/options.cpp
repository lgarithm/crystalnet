#include <getopt.h>

#include <experimental/filesystem>

#include <crystalnet-contrib/yolo/options.hpp>

namespace fs = std::experimental::filesystem;

options_t::options_t()
{
    const auto home = fs::path(std::getenv("HOME"));
    model_dir = home / "var/models/yolo";
    darknet_path = home / "code/mirrors/github.com/pjreddie/darknet";
    filename = darknet_path / "data/dog.jpg";
}

options_t parse_flags(int argc, char *argv[])
{
    options_t option;
    for (int c; (c = getopt(argc, argv, "f:m:")) != -1;) {
        switch (c) {
        case 'f':
            option.filename = optarg;
            break;
        case 'm':
            option.model_dir = optarg;
            break;
        default:
            fprintf(stderr, "unknown options: %c", c);
            break;
        }
    }
    return option;
}
