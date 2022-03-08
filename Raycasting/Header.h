#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

typedef double Float;
typedef unsigned int uint;
typedef unsigned long long ulong;
constexpr uint list_elem_empty_flag = -1;

//constexpr Float timestep = Float(0.005);
//constexpr Float ray_density = 100000;
//extern Float d_h;
inline static Float d_h = 1e-2;