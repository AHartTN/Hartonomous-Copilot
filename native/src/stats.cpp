/**
 * Statistical functions for meta-learning
 */

#include "../include/hartonomous_native.h"
#include <cmath>

extern "C" {

void welford_update(
    double old_mean,
    double old_m2,
    int old_count,
    double new_value,
    double* out_mean,
    double* out_m2)
{
    if (!out_mean || !out_m2 || old_count < 0) {
        return;
    }

    // Welford's online algorithm for numerically stable variance
    int new_count = old_count + 1;
    double delta = new_value - old_mean;
    double new_mean = old_mean + delta / new_count;
    double delta2 = new_value - new_mean;
    double new_m2 = old_m2 + delta * delta2;

    *out_mean = new_mean;
    *out_m2 = new_m2;
}

} // extern "C"
