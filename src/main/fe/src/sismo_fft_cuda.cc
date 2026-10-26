#include <cuda_runtime_api.h>
#include <cufft.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void checkCuda(cudaError_t err, const char* where)
{
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(err));
}

void checkCuFFT(cufftResult err, const char* where)
{
  if (err != CUFFT_SUCCESS)
    throw std::runtime_error(std::string(where) +
                             ": cuFFT error code " +
                             std::to_string(static_cast<int>(err)));
}
}

void computeSismoFFTcuFFT(const std::vector<float>& signals,
                          int nReceivers,
                          int nSamples,
                          float dt,
                          const std::string& fftFile)
{
  if (nReceivers <= 0 || nSamples <= 1)
    throw std::runtime_error("Invalid sismogram dimensions for cuFFT.");

  if (dt <= 0.0f)
    throw std::runtime_error("dt must be > 0 for cuFFT.");

  const std::size_t expected =
      static_cast<std::size_t>(nReceivers) *
      static_cast<std::size_t>(nSamples);

  if (signals.size() != expected)
    throw std::runtime_error("computeSismoFFTcuFFT: unexpected input size.");

  const int nFreq = nSamples / 2 + 1;

  const std::size_t inputBytes =
      expected * sizeof(cufftReal);

  const std::size_t outputCount =
      static_cast<std::size_t>(nReceivers) *
      static_cast<std::size_t>(nFreq);

  const std::size_t outputBytes =
      outputCount * sizeof(cufftComplex);

  std::vector<cufftReal> hostInput(expected);

  for (int r = 0; r < nReceivers; ++r)
  {
    const std::size_t base =
        static_cast<std::size_t>(r) * nSamples;

    double mean = 0.0;

    for (int t = 0; t < nSamples; ++t)
      mean += signals[base + t];

    mean /= static_cast<double>(nSamples);

    for (int t = 0; t < nSamples; ++t)
    {
      hostInput[base + t] =
          static_cast<cufftReal>(
              signals[base + t] - mean);
    }
  }

  cufftReal* dInput = nullptr;
  cufftComplex* dOutput = nullptr;
  cufftHandle plan{};

  checkCuda(
      cudaMalloc(
          reinterpret_cast<void**>(&dInput),
          inputBytes),
      "cudaMalloc(dInput)");

  try
  {
    checkCuda(
        cudaMalloc(
            reinterpret_cast<void**>(&dOutput),
            outputBytes),
        "cudaMalloc(dOutput)");

    checkCuFFT(
        cufftPlan1d(
            &plan,
            nSamples,
            CUFFT_R2C,
            nReceivers),
        "cufftPlan1d");

    const auto totalStart =
        std::chrono::steady_clock::now();

    const auto h2dStart =
        std::chrono::steady_clock::now();

    checkCuda(
        cudaMemcpy(
            dInput,
            hostInput.data(),
            inputBytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");

    const auto h2dEnd =
        std::chrono::steady_clock::now();

    const auto fftStart =
        std::chrono::steady_clock::now();

    checkCuFFT(
        cufftExecR2C(
            plan,
            dInput,
            dOutput),
        "cufftExecR2C");

    checkCuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize after cuFFT");

    const auto fftEnd =
        std::chrono::steady_clock::now();

    std::vector<cufftComplex> hostOutput(outputCount);

    const auto d2hStart =
        std::chrono::steady_clock::now();

    checkCuda(
        cudaMemcpy(
            hostOutput.data(),
            dOutput,
            outputBytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H");

    const auto d2hEnd =
        std::chrono::steady_clock::now();

    const auto totalEnd = d2hEnd;

    const double h2dMs =
        std::chrono::duration<double, std::milli>(
            h2dEnd - h2dStart)
            .count();

    const double fftMs =
        std::chrono::duration<double, std::milli>(
            fftEnd - fftStart)
            .count();

    const double d2hMs =
        std::chrono::duration<double, std::milli>(
            d2hEnd - d2hStart)
            .count();

    const double totalMs =
        std::chrono::duration<double, std::milli>(
            totalEnd - totalStart)
            .count();

    std::ofstream out(fftFile);

    if (!out)
      throw std::runtime_error(
          "Cannot open FFT output file: " + fftFile);

    out << std::setprecision(9);
    out << "receiver,frequency_hz,amplitude\n";

    for (int r = 0; r < nReceivers; ++r)
    {
      const std::size_t base =
          static_cast<std::size_t>(r) * nFreq;

      for (int k = 0; k < nFreq; ++k)
      {
        const cufftComplex z =
            hostOutput[base + k];

        const double amplitude =
            std::sqrt(
                static_cast<double>(z.x) * z.x +
                static_cast<double>(z.y) * z.y) *
            static_cast<double>(dt);

        const double frequency =
            static_cast<double>(k) /
            (static_cast<double>(nSamples) *
             static_cast<double>(dt));

        out << r << ','
            << frequency << ','
            << amplitude << '\n';
      }
    }

    std::ofstream timing(
        fftFile + ".timings.csv");

    if (!timing)
      throw std::runtime_error(
          "Cannot open FFT timing file: " +
          fftFile + ".timings.csv");

    timing << std::setprecision(9);

    timing
        << "n_receivers,n_samples,bytes_h2d,bytes_d2h,"
           "h2d_ms,fft_ms,d2h_ms,total_ms\n";

    timing << nReceivers << ','
           << nSamples << ','
           << inputBytes << ','
           << outputBytes << ','
           << h2dMs << ','
           << fftMs << ','
           << d2hMs << ','
           << totalMs << '\n';

    std::cout
        << "[In-situ cuFFT] receivers="
        << nReceivers
        << " samples="
        << nSamples
        << " H2D="
        << h2dMs
        << " ms"
        << " FFT="
        << fftMs
        << " ms"
        << " D2H="
        << d2hMs
        << " ms"
        << " total="
        << totalMs
        << " ms"
        << '\n';

    cufftDestroy(plan);
    plan = {};

    cudaFree(dOutput);
    dOutput = nullptr;

    cudaFree(dInput);
    dInput = nullptr;
  }
  catch (...)
  {
    if (plan)
      cufftDestroy(plan);

    if (dOutput)
      cudaFree(dOutput);

    if (dInput)
      cudaFree(dInput);

    throw;
  }
}
