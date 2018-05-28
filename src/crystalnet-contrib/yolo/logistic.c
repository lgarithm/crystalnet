#include <math.h>

#include <crystalnet-contrib/yolo/logistic.h>

float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
