#pragma once
#include <data_type.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tp3_slice.hpp"

struct TP3MapMetrics
{
  double update_s = 0, copy_s = 0, write_s = 0, render_s = 0, fsync_s = 0;
  std::uintmax_t bytes = 0, memory_bytes = 0, reference_memory_bytes = 0;
};

class TP3Maps
{
 public:
  TP3Maps(int nodes, const std::string& mode, float threshold,
          int reference_interval);
  void update(const arrayReal& pressure, int column, int step);
  void save(const std::string& folder, const std::vector<float>& coordinates,
            int steps, float dt, bool durable, TP3Slice* renderer);
  const TP3MapMetrics& metrics() const { return metrics_; }

 private:
  using Space = Kokkos::DefaultExecutionSpace::memory_space;
  int nodes_, reference_interval_;
  float threshold_;
  bool amplitude_, arrival_, energy_;
  Kokkos::View<float*, Space> maximum_, sampled_maximum_;
  Kokkos::View<int*, Space> first_, sampled_first_;
  Kokkos::View<double*, Space> energy_sum_, sampled_energy_, held_squared_;
  TP3MapMetrics metrics_;
};
