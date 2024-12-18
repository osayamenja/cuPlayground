//
// Created by oja7 on 12/18/24.
//

#ifndef UTIL_CUH
#define UTIL_CUH

#include <typeinfo>
#include <cxxabi.h>
#include <cuda_runtime.h>

#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))
#define BYTE_CAST(p) static_cast<cuda::std::byte*>(static_cast<void*>(p))
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#define TO_MB(b) static_cast<double>(b) / (1024.0f*1024.0f)
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#if !defined(CHECK_ERROR_EXIT)
#  define CHECK_ERROR_EXIT(e)                                         \
do {                                                           \
cudaError_t code = (e);                                      \
if (code != cudaSuccess) {                                   \
fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",               \
__FILE__, __LINE__, #e,                            \
cudaGetErrorName(code), cudaGetErrorString(code)); \
fflush(stderr);                                            \
exit(1);                                                   \
}                                                            \
} while (0)
#endif

#if !defined(CHECK_LAST)
# define CHECK_LAST() CHECK_ERROR_EXIT(cudaPeekAtLastError()); CHECK_ERROR_EXIT(cudaDeviceSynchronize())
#endif

__device__ __forceinline__
bool isLikelyRegister(void* p) {
    return !(__isShared(p) &&
        __isLocal(p) &&
        __isConstant(p) &&
        __isGlobal(p) &&
        __isGridConstant(p));
}

template<typename T>
void printType() {
    // Get the mangled name
    const char* mangledName = typeid(T).name();

    // Demangle the name
    int status;
    char* demangledName = abi::__cxa_demangle(mangledName, nullptr, nullptr, &status);

    // Print the demangled name
    if (status == 0) {
        std::cout << "Demangled name: " << demangledName << std::endl;
    } else {
        std::cerr << "Demangling failed!" << std::endl;
    }
    // Free the memory allocated by abi::__cxa_demangle
    free(demangledName);
}
#endif //UTIL_CUH
