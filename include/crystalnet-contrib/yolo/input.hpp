#pragma once
#include <stdtracer>

#include <crystalnet-contrib/vis/vis.hpp>
#include <crystalnet-contrib/yolo/detection.hpp>
#include <crystalnet-contrib/yolo/input.hpp>
#include <crystalnet-contrib/yolo/options.hpp>
#include <crystalnet-contrib/yolo/yolo.h>
#include <crystalnet-ext.h>
#include <crystalnet/debug/debug.hpp>
#include <crystalnet/model/model.hpp>

extern const tensor_t *load_test_image(const char * /* filename */);
extern const tensor_t *load_resized_test_image(const char * /* filename */);

extern void load_parameters(const model_t *,
                            const std::experimental::filesystem::path &);
extern std::vector<std::string>
load_name_list(const std::experimental::filesystem::path &);

extern std::string extension(const std::experimental::filesystem::path &);

template <typename T> tensor_t *hwc_to_chw(const tensor_ref_t &_t)
{
    const auto t = ranked<3, T>(_t);
    const auto [h, w, c] = t.shape.dims;
    tensor_t *__t = new tensor_t(shape_t(c, h, w), idx_type<T>::type);
    const auto r = ranked<3, T>(ref(*__t));
    for (auto l : range(c)) {
        for (auto i : range(h)) {
            for (auto j : range(w)) { r.at(l, i, j) = t.at(i, j, l); }
        }
    }
    return __t;
}

template <typename T> tensor_t *chw_to_hwc(const tensor_ref_t &_t)
{
    const auto t = ranked<3, T>(_t);
    const auto [c, h, w] = t.shape.dims;
    tensor_t *__t = new tensor_t(shape_t(h, w, c), idx_type<T>::type);
    const auto r = ranked<3, T>(ref(*__t));
    for (auto l : range(c)) {
        for (auto i : range(h)) {
            for (auto j : range(w)) { r.at(i, j, l) = t.at(l, i, j); }
        }
    }
    return __t;
}
