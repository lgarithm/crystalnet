#include "options.hpp"

#include <getopt.h>

namespace fs = std::experimental::filesystem;

options_t::options_t()
{
    const fs::path home(std::getenv("HOME"));
    const auto darknet_path = home / "code/mirrors/github.com/pjreddie/darknet";
    model_dir = home / "var/models/yolo";
    model_cfg = darknet_path / "cfg/yolov2.cfg";
    weights_file = model_dir / "yolov2.weights";
}

options_t parse_flags(int argc, char *argv[])
{
    options_t option;
    for (int c; (c = getopt(argc, argv, "c:")) != -1;) {
        switch (c) {
        case 'c':
            option.model_cfg = optarg;
            break;
        default:
            fprintf(stderr, "unknown options: %c", c);
            break;
        }
    }
    return option;
}
