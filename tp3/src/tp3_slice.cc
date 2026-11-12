#include "tp3_slice.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>

int TP3Slice::checkedPixelCount(int width, int height)
{
  if (width < 2 || height < 2 ||
      width > std::numeric_limits<int>::max() / height ||
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) >
          static_cast<std::size_t>(std::numeric_limits<int>::max() / 3))
  {
    throw std::invalid_argument("Resolution de la slice invalide");
  }
  return width * height;
}

TP3Slice::TP3Slice(int width, int height, const SliceUtils::PixelGrid& grid,
                   const std::string& backend, const std::string& normalization,
                   int window)
    : width_(width),
      height_(height),
      window_(window),
      plane_coordinate_(grid.coordinate),
      backend_(backend),
      normalization_(normalization),
      node_indices_("tp3_slice_node_indices", checkedPixelCount(width, height))
{
  if (backend_ != "cpu" && backend_ != "gpu")
    throw std::invalid_argument("Backend image invalide");
  if (backend_ == "gpu" &&
      std::string(Kokkos::DefaultExecutionSpace::name()) != "Cuda")
    throw std::invalid_argument("La generation GPU exige le backend CUDA");
  for (int n = 0; n < width_ * height_; ++n) node_indices_(n) = grid.nodes[n];
  std::set<int> unique;
  for (int n = 0; n < width_ * height_; ++n) unique.insert(node_indices_(n));
  unique_nodes_ = unique.size();
  if (backend_ == "gpu")
  {
    device_indices_ =
        decltype(device_indices_)("tp3_pixel_indices", width_ * height_);
    device_samples_ =
        decltype(device_samples_)("tp3_samples_gpu", width_ * height_);
    device_pixels_ =
        decltype(device_pixels_)("tp3_rgb_gpu", 3 * width_ * height_);
    Kokkos::deep_copy(device_indices_, node_indices_);
    Kokkos::fence();
  }
}

std::vector<std::uint8_t> TP3Slice::toRGB(const std::vector<float>& values,
                                          float fixed_scale) const
{
  const int count = width_ * height_;
  if (values.size() != static_cast<std::size_t>(count))
  {
    throw std::invalid_argument("Taille de slice incorrecte");
  }
  if (!std::isfinite(fixed_scale) || fixed_scale < 0.0f)
  {
    throw std::invalid_argument("Echelle de normalisation invalide");
  }

  Kokkos::View<float*, Kokkos::HostSpace> samples("tp3_slice_values", count);
  Kokkos::View<std::uint8_t*, Kokkos::HostSpace> pixels("tp3_slice_rgb",
                                                        3 * count);

  for (int i = 0; i < count; ++i)
  {
    const float value = values[static_cast<std::size_t>(i)];
    if (!std::isfinite(value))
    {
      throw std::invalid_argument("La slice contient une valeur non finie");
    }
    samples(i) = value;
  }

  float max_abs = 0.0f;
  if (fixed_scale == 0)
    Kokkos::parallel_reduce(
        "tp3_slice_max_abs_cpu",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, count),
        [=](const int index, float& local_max) {
          const float x = samples(index);
          const float magnitude = x < 0.0f ? -x : x;
          if (magnitude > local_max)
          {
            local_max = magnitude;
          }
        },
        Kokkos::Max<float>(max_abs));

  const float scale =
      fixed_scale > 0.0f ? fixed_scale : (max_abs > 0.0f ? max_abs : 1.0f);
  Kokkos::parallel_for(
      "tp3_slice_rgb_cpu",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, count),
      [=](const int index) {
        float normalized = samples(index) / scale;
        normalized = normalized < -1.0f
                         ? -1.0f
                         : (normalized > 1.0f ? 1.0f : normalized);
        const float magnitude = normalized < 0.0f ? -normalized : normalized;
        const std::uint8_t pale =
            static_cast<std::uint8_t>(255.0f * (1.0f - magnitude) + 0.5f);
        pixels(3 * index) = normalized < 0.0f ? pale : 255;
        pixels(3 * index + 1) = pale;
        pixels(3 * index + 2) = normalized > 0.0f ? pale : 255;
      });
  Kokkos::fence();

  std::vector<std::uint8_t> rgb(static_cast<std::size_t>(count) * 3);
  for (std::size_t i = 0; i < rgb.size(); ++i)
  {
    rgb[i] = pixels(static_cast<int>(i));
  }
  return rgb;
}

int TP3Slice::width() const { return width_; }

int TP3Slice::height() const { return height_; }

float TP3Slice::chooseScale(float maximum, float fixed)
{
  if (fixed > 0) return fixed;
  if (normalization_ == "running")
  {
    maxima_.push_back(maximum);
    if (window_ > 0 && maxima_.size() > std::size_t(window_))
      maxima_.pop_front();
    maximum = *std::max_element(maxima_.begin(), maxima_.end());
  }
  return maximum > 0 ? maximum : 1.f;
}
