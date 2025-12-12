/**
 * Version information
 */

#include "../include/hartonomous_native.h"

#define HARTONOMOUS_VERSION "1.0.0"
#define HARTONOMOUS_BUILD 1

extern "C" {

const char* hartonomous_version() {
    return HARTONOMOUS_VERSION;
}

int hartonomous_build_number() {
    return HARTONOMOUS_BUILD;
}

} // extern "C"
