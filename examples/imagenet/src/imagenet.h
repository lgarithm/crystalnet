#pragma once
#include <crystalnet-ext.h>

#include <opencv2/opencv.hpp>

cv::Mat square_normalize(const cv::Mat &, int r);
void info(const cv::Mat &, const std::string &);
std::size_t to_hwc(const cv::Mat &, void *);
