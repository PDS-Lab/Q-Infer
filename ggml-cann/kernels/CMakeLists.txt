if (NOT SOC_TYPE)
    set (SOC_TYPE "Ascend910B1")
endif()

file(GLOB SRC_FILES
    get_row_f32.cpp
    get_row_f16.cpp
    get_row_q4_0.cpp
    get_row_q8_0.cpp
    quantize_f32_q8_0.cpp
    quantize_f16_q8_0.cpp
    quantize_float_to_q4_0.cpp
    sparse_mul_mat_f16.cpp
    sparse_axpy_f16.cpp
    dup.cpp
)

string(TOLOWER ${SOC_TYPE} SOC_VERSION)
message(STATUS "SOC_VERSION: ${SOC_VERSION}")
set(ASCEND_CANN_PACKAGE_PATH ${CANN_INSTALL_DIR})
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")
set(RUN_MODE "npu" CACHE STRING "run mode: npu/sim")
message(STATUS "RUN_MODE: ${RUN_MODE}")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the compiler package is installed.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels STATIC
    ${SRC_FILES}
)

# ascendc_compile_definitions(ascendc_kernels PRIVATE -DASCENDC_DUMP)
