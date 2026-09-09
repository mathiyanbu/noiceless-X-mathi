cmake_minimum_required(VERSION 3.20)
project(SIH26052_NOICELESSX
    VERSION 0.1.0
    DESCRIPTION "Real-time dual-microphone speech enhancement and adaptive noise cancellation for Raspberry Pi"
    LANGUAGES C CXX
)

# Standard C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform & Architecture Tuning (Target: Raspberry Pi 4/5 ARM64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Detected target architecture: ARM64 (Raspberry Pi 4/5)")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-march=armv8-a+simd -O3 -fPIC)
    endif()
else()
    message(STATUS "Host architecture: ${CMAKE_SYSTEM_PROCESSOR} (Development / Host mode)")
endif()

# Common compilation flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
elseif(MSVC)
    add_compile_options(/W4 /EHsc /utf-8)
endif()

# Configuration Options
# Enable/disable ALSA (defaults ON on Linux, OFF on Windows/macOS)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    option(ENABLE_ALSA "Enable ALSA audio capture and playback backend" ON)
else()
    option(ENABLE_ALSA "Enable ALSA audio capture and playback backend" OFF)
endif()

# Enable/disable ONNX Runtime C++ API
option(ENABLE_ONNXRUNTIME "Enable ONNX Runtime CPU EP inference engine" ON)

# Enable/disable Test Suite
option(BUILD_TESTS "Build GoogleTest test suite" ON)

# Hardware Detection / ALSA Configuration
if(ENABLE_ALSA)
    find_package(ALSA)
    if(ALSA_FOUND)
        message(STATUS "ALSA found: ${ALSA_VERSION_STRING} (Library: ${ALSA_LIBRARIES})")
        add_compile_definitions(NOICELESSX_HAS_ALSA=1)
    else()
        message(WARNING "ALSA not found! Audio backend disabled. (Note: Runtime on Raspberry Pi requires real ALSA hardware and refuses READY without it.)")
        set(ENABLE_ALSA OFF)
        add_compile_definitions(NOICELESSX_HAS_ALSA=0)
    endif()
else()
    message(STATUS "ALSA backend disabled by configuration (Dev / Host mode).")
    add_compile_definitions(NOICELESSX_HAS_ALSA=0)
endif()

# ONNX Runtime Configuration
if(ENABLE_ONNXRUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        PATHS
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime
        $ENV{ONNXRUNTIME_ROOT}/include
    )
    find_library(ONNXRUNTIME_LIB NAMES onnxruntime
        PATHS
        /usr/lib
        /usr/local/lib
        $ENV{ONNXRUNTIME_ROOT}/lib
    )

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
        add_compile_definitions(NOICELESSX_HAS_ONNXRUNTIME=1)
    else()
        message(STATUS "ONNX Runtime not found in system paths. Header/interface mode enabled for dev host.")
        add_compile_definitions(NOICELESSX_HAS_ONNXRUNTIME=0)
    endif()
else()
    message(STATUS "ONNX Runtime disabled by configuration.")
    add_compile_definitions(NOICELESSX_HAS_ONNXRUNTIME=0)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/embedded
    ${CMAKE_CURRENT_SOURCE_DIR}/config
)

# Core library interface scaffolding
add_library(noiselessx_core INTERFACE)
target_include_directories(noiselessx_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/embedded
)

# Audio Subsystem Library
set(AUDIO_SOURCES
    embedded/audio/device_enumerator.cpp
    embedded/audio/alsa_backend.cpp
)

add_library(noiselessx_audio STATIC ${AUDIO_SOURCES})
target_include_directories(noiselessx_audio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/embedded
)
target_link_libraries(noiselessx_audio PUBLIC noiselessx_core)

if(ENABLE_ALSA AND ALSA_FOUND)
    target_link_libraries(noiselessx_audio PUBLIC ${ALSA_LIBRARIES})
    target_include_directories(noiselessx_audio PUBLIC ${ALSA_INCLUDE_DIRS})
endif()

if(ENABLE_ONNXRUNTIME AND ONNXRUNTIME_LIB)
    target_link_libraries(noiselessx_core INTERFACE ${ONNXRUNTIME_LIB})
    target_include_directories(noiselessx_core INTERFACE ${ONNXRUNTIME_INCLUDE_DIR})
endif()

# Hardware Test Utility (built when ALSA is enabled or on Linux target)
if(ENABLE_ALSA)
    add_executable(audio_test tools/audio_test.cpp)
    target_link_libraries(audio_test PRIVATE noiselessx_audio noiselessx_core)
endif()

# Testing Scaffolding with GoogleTest
if(BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Unit test executable covering lock-free ring buffers, drift detection, and sanity
    add_executable(noiselessx_unit_tests
        tests/unit/test_sanity.cpp
        tests/unit/test_ring_buffer.cpp
        tests/unit/test_drift_detector.cpp
    )

    target_link_libraries(noiselessx_unit_tests PRIVATE
        noiselessx_audio
        noiselessx_core
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(noiselessx_unit_tests)
endif()
