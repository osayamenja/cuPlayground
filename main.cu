#include <iostream>

#include <cuda/std/cassert>
#include <cuda/std/array>
#include <fmt/ranges.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h> // Makes CLion happy

#include <cuda/experimental/device.cuh>
#include "examples/pipelining.cuh"
#include "util.cuh"

int main() {
    runScheduler<SchedulerType::vanilla>();
    //runScheduler<SchedulerType::fast>();
}