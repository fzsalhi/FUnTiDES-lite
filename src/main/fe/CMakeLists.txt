if(COMPILE_SEM)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  option(SEMPROXY_ENABLE_CUFFT "Enable in-situ sismo FFT with cuFFT" OFF)
  add_executable(semproxy
    src/main.cc
    src/sem_proxy.cc
  )
  if(SEMPROXY_ENABLE_CUFFT)
    find_package(CUDAToolkit REQUIRED)

    target_sources(semproxy
      PRIVATE
        src/sismo_fft_cuda.cc
    )

    target_compile_definitions(semproxy
      PRIVATE
        SEMPROXY_ENABLE_CUFFT
    )

    target_link_libraries(semproxy
      PRIVATE
        CUDA::cufft_static_nocallback
        CUDA::cudart_static
    )
  endif()

  target_include_directories(semproxy
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}/external/cxxopts
  )

  target_link_libraries(semproxy
    PRIVATE
      discretization
      proxy_solver
      proxy_model_builder_cartesian
      proxy_model_struct
      proxy_model_unstruct
      proxy_utils
      ${STDCXXFS_LIB}
  )

install(TARGETS semproxy
        RUNTIME DESTINATION bin)

endif()
