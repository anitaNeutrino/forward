# this minimum version hasn't been
# tested extensively
cmake_minimum_required(VERSION 2.8.10)

# setup our project
project(WaveletDeconvolution
  LANGUAGES CXX)

# and a version
set(CMAKE_PROJECT_VERSION 0.1.0)

# create a static library
add_library(waveletdeconv INTERFACE)

# install
target_include_directories(waveletdeconv INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/>
)

# we also need access to ANITA tools to build waveletdeconv
target_include_directories(waveletdeconv PRIVATE "${ANITA_UTIL_INSTALL_DIR}/include")

# and we need ROOT
find_package(ROOT REQUIRED)

# and we need FFTW as well
find_package(FFTW3 REQUIRED)

# include against forward
target_include_directories(waveletdeconv PUBLIC "${PROJECT_SOURCE_DIR}/include")

set(COMPILE_OPTIONS -Wall -Wextra -Wdisabled-optimization -fconcepts
  -pedantic -Wshadow -Wno-sign-conversion -Wno-sign-compare -Wold-style-cast
  -Wno-strict-overflow -Wundef -Wno-unused -Wstack-protector -Wcast-align
  -Wpacked -Wformat-y2k -Warray-bounds -Wreorder -Wmissing-braces
  -Wstrict-overflow=4 -Wwrite-strings -Wpointer-arith
  -Wcast-qual -Wswitch-enum -Wswitch-default -Wunreachable-code
  -Wmaybe-uninitialized -march=native -fmax-errors=5 -O3 -Wfloat-equal)

# add the compile options to the library
target_compile_options(waveletdeconv PRIVATE ${COMPILE_OPTIONS})

# setup our public headers
set_target_properties(waveletdeconv PROPERTIES PUBLIC_HEADER "WaveletDeconvolution.h")

# and install the headers and the library
INSTALL(TARGETS waveletdeconv
        PUBLIC_HEADER DESTINATION include)
