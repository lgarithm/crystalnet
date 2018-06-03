#include <math.h>

#include <crystalnet-contrib/yolo/cmath.h>

float c_exp(float x) { return exp(x); }
float c_logistic(float x) { return 1. / (1. + exp(-x)); }
