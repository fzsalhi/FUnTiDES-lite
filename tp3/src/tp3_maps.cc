#include "tp3_maps.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

#include "sem_proxy.h"

static_assert(sizeof(float) == 4 && sizeof(int) == 4 && sizeof(double) == 8,
              "Format des cartes incompatible avec les types scalaires");

TP3Maps::TP3Maps(int nodes, const std::string& mode, float threshold,
                 int reference_interval)
    : nodes_(nodes),
      reference_interval_(reference_interval),
      threshold_(threshold),
      amplitude_(mode == "all" || mode == "amplitude"),
      arrival_(mode == "all" || mode == "arrival"),
      energy_(mode == "all" || mode == "energy")
{
  if (amplitude_)
  {
    maximum_ = decltype(maximum_)("map_maximum", nodes);
    Kokkos::deep_copy(maximum_, 0.f);
  }
  if (arrival_)
  {
    first_ = decltype(first_)("map_arrival", nodes);
    Kokkos::deep_copy(first_, -1);
  }
  if (energy_)
  {
    energy_sum_ = decltype(energy_sum_)("map_energy", nodes);
    Kokkos::deep_copy(energy_sum_, 0.);
  }
  metrics_.memory_bytes =
      std::uintmax_t(nodes) * (4 * amplitude_ + 4 * arrival_ + 8 * energy_);
  if (reference_interval_ > 0)
  {
    if (amplitude_)
    {
      sampled_maximum_ = decltype(sampled_maximum_)("sampled_maximum", nodes);
      Kokkos::deep_copy(sampled_maximum_, 0.f);
    }
    if (arrival_)
    {
      sampled_first_ = decltype(sampled_first_)("sampled_arrival", nodes);
      Kokkos::deep_copy(sampled_first_, -1);
    }
    if (energy_)
    {
      sampled_energy_ = decltype(sampled_energy_)("sampled_energy", nodes);
      held_squared_ = decltype(held_squared_)("sampled_energy_hold", nodes);
      Kokkos::deep_copy(sampled_energy_, 0.);
      Kokkos::deep_copy(held_squared_, 0.);
    }
    metrics_.reference_memory_bytes =
        std::uintmax_t(nodes) * (4 * amplitude_ + 4 * arrival_ + 16 * energy_);
  }
  Kokkos::fence();
}

void TP3Maps::update(const arrayReal& pressure, int column, int step)
{
  Kokkos::fence();
  const auto start = tp1::Clock::now();
  auto maximum = maximum_;
  auto first = first_;
  auto sum = energy_sum_;
  auto sm = sampled_maximum_;
  auto sf = sampled_first_;
  auto se = sampled_energy_;
  auto hold = held_squared_;
  const bool amplitude = amplitude_, arrival = arrival_, energy = energy_,
             reference = reference_interval_ > 0;
  const bool sampled = reference && step % reference_interval_ == 0;
  const float threshold = threshold_;
  Kokkos::parallel_for(
      "tp3_update_maps", nodes_, KOKKOS_LAMBDA(int n) {
        const float p = pressure(n, column), magnitude = p < 0 ? -p : p;
        if (amplitude && magnitude > maximum(n)) maximum(n) = magnitude;
        if (arrival && first(n) < 0 && magnitude > threshold) first(n) = step;
        if (energy)
        {
          const double square = double(p) * double(p);
          sum(n) += square;
          if (reference)
          {
            if (sampled) hold(n) = square;
            se(n) += hold(n);
          }
        }
        if (reference)
        {
          if (sampled)
          {
            if (amplitude && magnitude > sm(n)) sm(n) = magnitude;
            if (arrival && sf(n) < 0 && magnitude > threshold) sf(n) = step;
          }
        }
      });
  Kokkos::fence();
  metrics_.update_s += tp1::seconds(start);
}

namespace
{
template <class T>
std::vector<T> hostValues(
    const Kokkos::View<T*, Kokkos::DefaultExecutionSpace::memory_space>& values)
{
  auto mirror =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
  Kokkos::fence();
  std::vector<T> result(values.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = mirror(i);
  return result;
}
template <class T>
void writeValues(std::ofstream& out, const std::vector<T>& values)
{
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian) == 1)
    out.write(reinterpret_cast<const char*>(values.data()),
              values.size() * sizeof(T));
  else
    for (T value : values)
    {
      char bytes[sizeof(T)];
      std::memcpy(bytes, &value, sizeof(T));
      std::reverse(bytes, bytes + sizeof(T));
      out.write(bytes, sizeof(T));
    }
}
}  // namespace

void TP3Maps::save(const std::string& folder,
                   const std::vector<float>& coordinates, int steps, float dt,
                   bool durable, TP3Slice* renderer)
{
  namespace fs = std::filesystem;
  fs::create_directories(folder);
  auto finish = [&](std::ofstream& out, const fs::path& path) {
    metrics_.bytes += tp1::finish(out, path, durable, metrics_.fsync_s);
  };
  auto start = tp1::Clock::now();
  const auto coords = fs::path(folder) / "coords.bin";
  {
    std::ofstream out(coords, std::ios::binary);
    writeValues(out, coordinates);
    finish(out, coords);
  }
  metrics_.write_s += tp1::seconds(start);
  std::ostringstream summary;
  summary << std::setprecision(17)
          << "quantity,nodes,memory_bytes,reference_interval,rmse_norm,max_"
             "error_norm,missed_nodes,reached_nodes,mean_delay_s,max_delay_s\n";
  auto saveField = [&](const std::string& name, const auto& device,
                       const auto& sampled, const char* dtype, bool arrival) {
    using View = std::decay_t<decltype(device)>;
    using T = typename View::non_const_value_type;
    auto phase = tp1::Clock::now();
    const auto values = hostValues(device);
    std::vector<T> reference;
    if (reference_interval_ > 0) reference = hostValues(sampled);
    metrics_.copy_s += tp1::seconds(phase);
    auto writeField = [&](const std::string& label, const std::vector<T>& field,
                          int interval) {
      phase = tp1::Clock::now();
      const auto base = fs::path(folder) / label;
      {
        const auto path = base.string() + ".bin";
        std::ofstream out(path, std::ios::binary);
        writeValues(out, field);
        finish(out, path);
      }
      const auto path = base.string() + ".json";
      std::ofstream out(path);
      out << std::setprecision(17)
          << "{\"schema\":\"funtides.map.v1\",\"quantity\":\"" << name
          << "\",\"dtype\":\"" << dtype << "\",\"count\":" << nodes_
          << ",\"coordinates\":\"coords.bin\",\"steps\":" << steps
          << ",\"dt_s\":" << dt << ",\"sample_interval\":" << interval
          << ",\"arrival_threshold\":" << threshold_
          << ",\"arrival_step_origin\":0,\"arrival_missing\":-1,\"energy\":"
             "\"sum(p*p), sampled: zero-order hold\"}\n";
      finish(out, path);
      metrics_.write_s += tp1::seconds(phase);
    };
    writeField(name, values, 1);
    if (reference_interval_ > 0)
      writeField(name + "_sampled", reference, reference_interval_);
    double scale = 0, squared_error = 0, max_error = 0, delay_sum = 0,
           max_delay = 0;
    std::size_t missed = 0, reached = 0, compared = 0;
    for (std::size_t n = 0; n < values.size(); ++n)
    {
      const double a = values[n];
      if (!std::isfinite(a))
        throw std::runtime_error("Carte contenant une valeur non finie");
      scale = std::max(scale, std::abs(a));
      if (arrival && a >= 0) ++reached;
      if (reference_interval_ > 0)
      {
        if (arrival && a >= 0 && reference[n] < 0)
        {
          ++missed;
          continue;
        }
        if (arrival && (a < 0 || reference[n] < 0)) continue;
        const double error = double(reference[n]) - a;
        squared_error += error * error;
        max_error = std::max(max_error, std::abs(error));
        ++compared;
        if (arrival)
        {
          delay_sum += error * dt;
          max_delay = std::max(max_delay, error * dt);
        }
      }
    }
    scale = scale > 0 ? scale : 1.;
    summary << name << ',' << nodes_ << ',' << nodes_ * sizeof(T) << ','
            << reference_interval_ << ',';
    if (reference_interval_ > 0)
      summary << std::sqrt(squared_error / std::max(std::size_t(1), compared)) /
                     scale
              << ',' << max_error / scale;
    else
      summary << ',';
    summary << ',' << missed << ',' << reached << ','
            << (compared ? delay_sum / compared : 0) << ',' << max_delay
            << '\n';
    if (renderer)
    {
      const auto renderStart = tp1::Clock::now();
      auto image = allocateArray2D<arrayReal>(nodes_, 1, "map_image_values");
      float imageScale = 0;
      for (int n = 0; n < nodes_; ++n)
      {
        const float x =
            arrival
                ? (values[n] < 0 ? 0.f : float((double(values[n]) + 1) * dt))
                : float(values[n]);
        image(n, 0) = x;
        imageScale = std::max(imageScale, x);
      }
      const auto frame =
          renderer->savePPM(image, 0, (fs::path(folder) / name).string(), "P6",
                            imageScale > 0 ? imageScale : 1.f, durable);
      for (const auto& file : frame.files)
      {
        metrics_.bytes += file.size_bytes;
        metrics_.fsync_s += file.fsync_ms / 1000;
      }
      metrics_.render_s += tp1::seconds(renderStart);
    }
  };
  if (amplitude_)
    saveField("maximum_amplitude", maximum_, sampled_maximum_, "<f4", false);
  if (arrival_) saveField("first_arrival", first_, sampled_first_, "<i4", true);
  if (energy_)
    saveField("cumulative_energy", energy_sum_, sampled_energy_, "<f8", false);
  start = tp1::Clock::now();
  const auto path = fs::path(folder) / "maps_summary.csv";
  std::ofstream out(path);
  out << summary.str();
  finish(out, path);
  metrics_.write_s += tp1::seconds(start);
}
