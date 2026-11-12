add_library(proxy_utils INTERFACE)

target_include_directories(proxy_utils
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src/utils/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/discretization/fe/SEMKernels/src>
)

add_library(proxy_compression STATIC ${CMAKE_SOURCE_DIR}/tp3/src/compression.cpp)
target_include_directories(proxy_compression PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tp3/include>)
target_compile_features(proxy_compression PUBLIC cxx_std_17)
target_link_libraries(proxy_utils INTERFACE $<BUILD_INTERFACE:proxy_compression>)
