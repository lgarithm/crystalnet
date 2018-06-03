#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <crystalnet-ext.h>
#include <crystalnet-internal.h>

#include <crystalnet-contrib/image/bmp.hpp>
#include <crystalnet-contrib/vis/snapshot.hpp>
#include <crystalnet-contrib/vis/vis.hpp>
#include <crystalnet-contrib/yolo/detection.hpp>
#include <crystalnet-contrib/yolo/input.hpp>
#include <crystalnet-contrib/yolo/yolo.h>
#include <crystalnet-contrib/yolo/yolo.hpp>
#include <crystalnet/core/tracer.hpp>  // TODO: don't include private headers
#include <crystalnet/debug/debug.hpp>
#include <crystalnet/model/model.hpp>

namespace fs = std::experimental::filesystem;

const fs::path home(std::getenv("HOME"));
const fs::path work_dir(std::getenv("PWD"));
const fs::path log_dir = work_dir / "log";

void run(const options_t &opt)
{
    const auto names = load_name_list(opt.darknet_path / "data/coco.names");
    const auto input = [&]() {
        if (extension(opt.filename) == ".bmp" ||
            extension(opt.filename) == ".idx") {
            // logf("using resized test image: %s", opt.filename.c_str());
            return load_resized_test_image(opt.filename.c_str());
        }
        return load_test_image(opt.filename.c_str());
    }();
    // logf("input image: %s", std::to_string(input->shape).c_str());

    const s_model_t *s_model = []() {
        TRACE("main::build symbolic model");
        return yolov2();
    }();
    {
        FILE *fp = fopen("graph.dot", "w");
        graphviz(*s_model, fp);
        fclose(fp);
        system("dot -Tsvg graph.dot -O");
    }
    const auto p_ctx = new_parameter_ctx();
    const model_t *p_model = [&]() {
        TRACE("main::build physical model");
        return realize(p_ctx, s_model, 1);
    }();
    {
        TRACE("main::load parameters");
        load_parameters(p_model, opt.model_dir);
    }
    {
        TRACE("main::inference");
        const auto r = ref(*input);
        p_model->input.bind(r.reshape(r.shape.batch(1)));
        p_model->forward();
    }
    {
        TRACE("main::print layers");
        SET_TRACE_LOG(log_dir / "cn-layers.txt");
        using T = float;
        logf("input %-3d %s", 0, summary(r_tensor_ref_t<T>(*input)).c_str());
        show_layers(*p_model, *s_model);
    }
    const auto dets = [&]() {
        using T = float;
        const auto r = ranked<4, T>(p_model->output.value());
        const auto [n, c, h, w] = r.shape.dims;
        const auto y = p_model->output.value().reshape(shape_t(c, h, w));
        const auto anchor_boxes = ref(*p_ctx->index.at("yolov2_31_anchors"));
        // logf("%s", summary(r_tensor_ref_t<T>(anchor_boxes)).c_str());
        return get_detections(y, anchor_boxes);
    }();
    {
        TRACE("main::print detections");
        SET_TRACE_LOG(log_dir / "cn-bboxes.txt");
        int i = 0;
        for (const auto &d : dets) {
            const auto b = d->bbox;
            logf("box %-4d: %s (%f, %f) [%f, %f]", i++,             //
                 summary(r_tensor_ref_t<float>(d->probs)).c_str(),  //
                 b.cx, b.cy, b.w, b.h);
        }
    }
    {
        TRACE("main::draw detections");
        using T = float;
        const auto n = input->shape.dim();
        const auto _x = chw_to_hwc<T>(ref(*input));
        const auto x = r_tensor_ref_t<T>(*_x);
        const tensor_t _y(x.shape, dtypes.u8);
        const auto y = ranked<3, uint8_t>(ref(_y));
        std::transform(x.data, x.data + n, y.data,
                       [](T v) { return (uint8_t)(v * 255); });
        for (const auto &d : dets) {
            const auto probs = ranked<1, T>(ref(d->probs));
            const auto idx =
                std::max_element(probs.data, probs.data + 80) - probs.data;
            if (d->scale > 0.55 && probs.at(idx) > 0.55) {
                logf("%-4d (%s): %f", idx, names.at(idx).c_str(),
                     probs.at(idx));
                const auto c =
                    rasterize(d->bbox, yolov2_input_size, yolov2_input_size);
                draw_clip(y, c);
            }
        }
        write_bmp_file("input.bmp", y);
    }

    del_model(p_model);
    del_s_model(s_model);
    del_tensor(input);
}

int main(int argc, char *argv[])
{
    TRACE(__func__);
    const auto opt = parse_flags(argc, argv);
    try {
        run(opt);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "unknown exception" << std::endl;
        return 2;
    }
    return 0;
}
