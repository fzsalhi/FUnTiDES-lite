#pragma once
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <slice_utils.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "tp3_ppm_visu.hpp"

struct TP3FrameMetrics
{
  double extract_ms = 0, color_ms = 0, copy_ms = 0, total_ms = 0;
  float max_abs = 0, scale = 1;
  std::size_t host_bytes = 0;
  std::vector<TP3PPMWriteMetrics> files;
};

class TP3Slice
{
 public:
  template <class Mesh>
  TP3Slice(Mesh& mesh, const int elements[3], const float lengths[3],
           const std::string& plane, float position, int width, int height,
           const std::string& backend = "cpu",
           const std::string& normalization = "local", int window = 8)
      : TP3Slice(width, height,
                 SliceUtils::uniformPixels(mesh, lengths, plane, position,
                                           width, height),
                 backend, normalization, window)
  {
    (void)elements;
  }

  template <class PressureView>
  std::vector<float> extract(const PressureView& pressure, int column) const
  {
    const int count = width_ * height_;
    Kokkos::View<float*, Kokkos::HostSpace> samples("tp3_samples_cpu", count);
    auto indices = node_indices_;
    SliceUtils::gather<Kokkos::DefaultHostExecutionSpace>(pressure, column,
                                                          indices, samples);
    Kokkos::fence();
    std::vector<float> values(count);
    for (int i = 0; i < count; ++i) values[i] = samples(i);
    return values;
  }

  std::vector<std::uint8_t> toRGB(const std::vector<float>& values,
                                  float fixed_scale = 0) const;

  template <class PressureView>
  TP3FrameMetrics savePPM(const PressureView& pressure, int column,
                          const std::string& prefix,
                          const std::string& format = "both",
                          float fixed_scale = 0, bool sync = false)
  {
    if (column < 0 || std::size_t(column) >= pressure.extent(1))
      throw std::out_of_range("Colonne de pression invalide");
    if (format != "P3" && format != "P6" && format != "both")
      throw std::invalid_argument("Format PPM invalide");
    using Clock = std::chrono::steady_clock;
    Kokkos::fence();
    const auto start = Clock::now();
    TP3FrameMetrics metrics;
    std::vector<std::uint8_t> rgb;
    const int count = width_ * height_;
    auto milliseconds = [](auto t) {
      return std::chrono::duration<double, std::milli>(Clock::now() - t)
          .count();
    };
    if (backend_ == "gpu")
    {
      auto indices = device_indices_;
      auto samples = device_samples_;
      auto source = pressure;
      SliceUtils::gather<Kokkos::DefaultExecutionSpace>(source, column, indices,
                                                        samples);
      Kokkos::fence();
      metrics.extract_ms = milliseconds(start);
      auto phase = Clock::now();
      float maximum = 0;
      Kokkos::parallel_reduce(
          "tp3_scale_gpu", count,
          KOKKOS_LAMBDA(int i, float& m) {
            const float x = samples(i);
            const float a = x < 0 ? -x : x;
            if (x != x)
              m = std::numeric_limits<float>::infinity();
            else if (a > m)
              m = a;
          },
          Kokkos::Max<float>(maximum));
      Kokkos::fence();
      if (!std::isfinite(maximum))
        throw std::runtime_error("Pression non finie pendant le rendu");
      metrics.max_abs = maximum;
      metrics.scale = chooseScale(maximum, fixed_scale);
      const float scale = metrics.scale;
      auto pixels = device_pixels_;
      Kokkos::parallel_for(
          "tp3_color_gpu", count, KOKKOS_LAMBDA(int i) {
            float x = samples(i) / scale;
            x = x < -1 ? -1 : (x > 1 ? 1 : x);
            const float a = x < 0 ? -x : x;
            const auto pale =
                static_cast<std::uint8_t>(255.f * (1.f - a) + .5f);
            pixels(3 * i) = x < 0 ? pale : 255;
            pixels(3 * i + 1) = pale;
            pixels(3 * i + 2) = x > 0 ? pale : 255;
          });
      Kokkos::fence();
      metrics.color_ms = milliseconds(phase);
      phase = Clock::now();
      auto host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, pixels);
      Kokkos::fence();
      rgb.resize(3 * std::size_t(count));
      for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = host(i);
      metrics.copy_ms = milliseconds(phase);
      metrics.host_bytes = rgb.size() + sizeof(float);
    }
    else
    {
      const auto values = extract(pressure, column);
      metrics.extract_ms = milliseconds(start);
      const auto phase = Clock::now();
      for (float x : values)
        metrics.max_abs = std::max(metrics.max_abs, std::abs(x));
      metrics.scale = chooseScale(metrics.max_abs, fixed_scale);
      rgb = toRGB(values, metrics.scale);
      metrics.color_ms = milliseconds(phase);
      metrics.host_bytes = unique_nodes_ * sizeof(float);
    }
    if (format == "P3" || format == "both")
      metrics.files.push_back(
          writePPM_P3(prefix + "_P3.ppm", width_, height_, rgb, sync));
    if (format == "P6" || format == "both")
      metrics.files.push_back(
          writePPM_P6(prefix + "_P6.ppm", width_, height_, rgb, sync));
    metrics.total_ms = milliseconds(start);
    return metrics;
  }
  int width() const;
  int height() const;
  float planeCoordinate() const { return plane_coordinate_; }
  const std::string& backend() const { return backend_; }

 private:
  TP3Slice(int width, int height, const SliceUtils::PixelGrid& grid,
           const std::string& backend, const std::string& normalization,
           int window);
  static int checkedPixelCount(int width, int height);
  float chooseScale(float maximum, float fixed);
  int width_, height_, window_;
  float plane_coordinate_;
  std::string backend_, normalization_;
  std::deque<float> maxima_;
  std::size_t unique_nodes_ = 0;
  Kokkos::View<int*, Kokkos::HostSpace> node_indices_;
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>
      device_indices_;
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space>
      device_samples_;
  Kokkos::View<std::uint8_t*, Kokkos::DefaultExecutionSpace::memory_space>
      device_pixels_;
};
