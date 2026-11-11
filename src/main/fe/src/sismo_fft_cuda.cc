#include "sem_proxy.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace {
void checkCuda(cudaError_t error, const char* where) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(error));
}
void checkCuFFT(cufftResult error, const char* where) {
  if (error != CUFFT_SUCCESS)
    throw std::runtime_error(std::string(where) + ": cuFFT error " + std::to_string(error));
}
}

tp1::FFTMetrics computeSismoFFTcuFFT(const arrayReal& traces, int receivers,
                                   int samples, float dt,
                                   const std::string& file, bool durable) {
  if (receivers <= 0 || samples <= 1 || dt <= 0)
    throw std::runtime_error("Invalid dimensions or timestep for cuFFT");
  Kokkos::fence();
  const auto totalStart = tp1::Clock::now();
  tp1::FFTMetrics timing;
  const int frequencies = samples / 2 + 1;
  const std::size_t inputCount = std::size_t(receivers) * samples;
  const std::size_t outputCount = std::size_t(receivers) * frequencies;
  cufftReal* input = nullptr;
  cufftComplex* output = nullptr;
  cufftHandle plan{};
  try {
    auto start = tp1::Clock::now();
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&input), inputCount * sizeof(cufftReal)), "cudaMalloc input");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&output), outputCount * sizeof(cufftComplex)), "cudaMalloc output");
    checkCuFFT(cufftPlan1d(&plan, samples, CUFFT_R2C, receivers), "cufftPlan1d");
    timing.plan_s = tp1::seconds(start);

    start = tp1::Clock::now();
#ifdef ENABLE_CUDA
    // Traces stay on the device; only the spectrum is returned to the host.
    {
      Kokkos::View<double*, Kokkos::CudaSpace> means("fft_means", receivers);
      Kokkos::parallel_for("fft_means", Kokkos::RangePolicy<>(0, receivers), KOKKOS_LAMBDA(int r) {
        double sum = 0;
        for (int t = 0; t < samples; ++t) sum += traces(r, t);
        means(r) = sum / samples;
      });
      Kokkos::parallel_for("fft_center_pack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
          Kokkos::IndexType<std::size_t>>(0, inputCount), KOKKOS_LAMBDA(std::size_t index) {
        const int r = index / samples, t = index % samples;
        input[index] = static_cast<float>(double(traces(r, t)) - means(r));
      });
      Kokkos::fence();
    }
    timing.prepare_s = tp1::seconds(start);
#else
    std::vector<cufftReal> host(inputCount);
    for (int r = 0; r < receivers; ++r) {
      double mean = 0;
      for (int t = 0; t < samples; ++t) mean += traces(r, t);
      mean /= samples;
      for (int t = 0; t < samples; ++t)
        host[std::size_t(r) * samples + t] = static_cast<float>(double(traces(r, t)) - mean);
    }
    timing.prepare_s = tp1::seconds(start);
    start = tp1::Clock::now();
    timing.h2d_bytes = inputCount * sizeof(cufftReal);
    checkCuda(cudaMemcpy(input, host.data(), timing.h2d_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    timing.h2d_s = tp1::seconds(start);
#endif

    start = tp1::Clock::now();
    checkCuFFT(cufftExecR2C(plan, input, output), "cufftExecR2C");
    checkCuda(cudaDeviceSynchronize(), "cuFFT synchronization");
    timing.compute_s = tp1::seconds(start);
    std::vector<cufftComplex> spectrum(outputCount);
    start = tp1::Clock::now();
    timing.d2h_bytes = outputCount * sizeof(cufftComplex);
    checkCuda(cudaMemcpy(spectrum.data(), output, timing.d2h_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    timing.d2h_s = tp1::seconds(start);

    start = tp1::Clock::now();
    const std::filesystem::path path(file);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open spectrum file: " + file);
    stream << std::setprecision(17) << "receiver,frequency_hz,amplitude\n";
    for (int r = 0; r < receivers; ++r)
      for (int k = 0; k < frequencies; ++k) {
        const auto z = spectrum[std::size_t(r) * frequencies + k];
        const double amplitude = std::hypot(double(z.x), double(z.y)) * double(dt);
        stream << r << ',' << double(k) / (double(samples) * double(dt)) << ',' << amplitude << '\n';
      }
    timing.output_bytes = tp1::finish(stream, path, durable, timing.fsync_s);
    timing.write_s = tp1::seconds(start);
    checkCuFFT(cufftDestroy(plan), "cufftDestroy"); plan = {};
    checkCuda(cudaFree(output), "cudaFree output"); output = nullptr;
    checkCuda(cudaFree(input), "cudaFree input"); input = nullptr;
    timing.total_s = tp1::seconds(totalStart);
    return timing;
  } catch (...) {
    if (plan) cufftDestroy(plan);
    if (output) cudaFree(output);
    if (input) cudaFree(input);
    throw;
  }
}
