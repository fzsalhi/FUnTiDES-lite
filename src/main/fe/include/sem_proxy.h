//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.hpp: the main interface of SEM proxy application
//
//************************************************************************
#ifndef SEMPROXY_HPP_
#define SEMPROXY_HPP_
#include <data_type.h>
#include <model_struct.h>
#include <model_unstruct.h>
#include <solver_factory.h>
#include <utils.h>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <fstream>
#include <limits>
#include <cmath>
#include <array>
#include "sem_proxy_options.h"
#include "compression_config.hpp"
#include "tp3_slice.hpp"
#include "tp3_maps.hpp"


#include <chrono>
#include <cstdint>
#include <filesystem>

namespace tp1 {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
std::uintmax_t finish(std::ofstream& stream, const std::filesystem::path& path,
                     bool durable, double& fsync_s);
struct FFTMetrics {
  double prepare_s = 0, plan_s = 0, h2d_s = 0, compute_s = 0;
  double d2h_s = 0, write_s = 0, total_s = 0, fsync_s = 0;
  std::uintmax_t h2d_bytes = 0, d2h_bytes = 0, output_bytes = 0;
};
struct Metrics {
  double compute_s = 0, receiver_sample_s = 0, snapshot_write_s = 0;
  double sismo_write_s = 0, other_processing_s = 0, fsync_s = 0, run_s = 0;
  std::uintmax_t snapshot_bytes = 0, sismo_bytes = 0;
  int snapshot_count = 0, completed_steps = 0;
  double stats_compute_s = 0, stats_copy_s = 0, stats_write_s = 0;
  double sismo_stats_s = 0, sismo_stats_copy_s = 0, sismo_stats_write_s = 0;
  double slice_setup_s = 0, slice_extract_s = 0, slice_copy_s = 0, slice_write_s = 0;
  std::uintmax_t stats_bytes = 0, sismo_stats_bytes = 0, slice_bytes = 0;
  std::uintmax_t stats_transfer_bytes = 0;
  int slice_count = 0;
  double ppm_setup_s=0, ppm_s=0, ppm_extract_s=0, ppm_color_s=0, ppm_copy_s=0;
  double ppm_encode_s=0, ppm_write_s=0, ppm_fsync_s=0;
  std::uintmax_t ppm_bytes=0, ppm_metrics_bytes=0, ppm_host_bytes=0;
  int ppm_frames=0, ppm_images=0;
  FFTMetrics fft;
};
}  // namespace tp1

#ifdef SEMPROXY_ENABLE_CUFFT
tp1::FFTMetrics computeSismoFFTcuFFT(const arrayReal& traces, int nReceivers,
                                   int nSamples, float dt,
                                   const std::string& fftFile, bool durable);
#endif

/* Streaming statistics (Welford's algorithm): constant memory*/

struct StreamStats
{
  long n = 0;
  double mean = 0.0;
  double m2 = 0.0;  // sum of squared deviations from the running mean
  double minVal = std::numeric_limits<double>::max();
  double maxVal = std::numeric_limits<double>::lowest();

  KOKKOS_INLINE_FUNCTION void add(double x)
  {
    n++;
    if (x < minVal) minVal = x;
    if (x > maxVal) maxVal = x;
    const double delta = x - mean;
    mean += delta / n;
    m2 += delta * (x - mean);
  }
  KOKKOS_INLINE_FUNCTION double variance() const { return n > 0 ? m2 / n : 0.0; }
  KOKKOS_INLINE_FUNCTION void merge(const StreamStats& b) {
    if (!b.n) return;
    if (!n) { *this = b; return; }
    const double delta = b.mean - mean;
    const long total = n + b.n;
    m2 += b.m2 + delta * delta * (double(n) * b.n / total);
    mean += delta * (double(b.n) / total);
    n = total;
    if (b.minVal < minVal) minVal = b.minVal;
    if (b.maxVal > maxVal) maxVal = b.maxVal;
  }
};

struct PressureReduction {
  using value_type = StreamStats;
  arrayReal pressure;
  int column;
  KOKKOS_INLINE_FUNCTION void init(value_type& value) const { value = StreamStats{}; }
  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const { dst.merge(src); }
  KOKKOS_INLINE_FUNCTION void operator()(const int n, value_type& value) const { value.add(pressure(n, column)); }
};


/**
 * @class SEMproxy
 */
class SEMproxy
{
 public:
 std::ofstream compress_out_;
  /**
   * @brief Constructor of the SEMproxy class
   */
  SEMproxy(const SemProxyOptions& cfg);
  /**
   * @brief Destructor of the SEMproxy class
   */
  ~SEMproxy(){};
  /**
   * @brief Initialize the simulation.
   * @post run()
   */
  void initFiniteElem()
  {
    init_arrays();
    init_source();
  };
  /**
   * @brief Run the simulation.
   * @pre This must be called after init()
   * @post Optional printout performance resutls
   */
  void run();
  /**
   * @brief Computes optimal time step using CFL stability condition for seismic
   * wave propagation
   *
   * Calculates the maximum stable time step for explicit finite difference
   * schemes using the CFL condition: dt ≤ CFL_factor × min_spacing / (√D ×
   * v_max) where D is the number of dimension.
   *
   * @param cfl_factor Stability factor (0.5-0.7 for 2nd-order, 0.3-0.4 for
   * higher-order schemes)
   * @return dt the max timestep.
   *
   * @note Typical values: 0.5-0.7 for 2nd-order, 0.3-0.4 for higher-order
   * schemes
   * @warning Must be called before time-stepping loop to ensure numerical
   * stability
   *
   */
  float find_cfl_dt(float cfl_factor);
  void saveSnapshot(int timestep);
  void sampleReceivers(int timestep);
  void updateReceiverStatistics(int timestep);
  void saveSlice(int timestep);
  void writeMetrics(double initial_s, double total_s) const;
  // In-situ statistics of one pressure column computed on the device with
  // Kokkos::parallel_reduce. It must be public: nvcc forbids extended
  // lambdas (KOKKOS_LAMBDA) inside private/protected member functions.
  void computePressureStatistics(int column, int step);
 private:
  int i1 = 0;
  int i2 = 1;
  CompressConfig compress_cfg_;
  // proper to cartesian mesh
  // or any structured mesh
  int nb_elements_[3] = {0};
  int nb_nodes_[3] = {0};
  std::array<float, 3> src_coord_ = {0};
  std::array<float, 3> rcv_coord_ = {0};
  // sismos : positions de TOUS les recepteurs ({x, y, z} chacun)
  std::vector<std::array<float, 3>> rcv_coords_;
  // sismos : nom du fichier CSV de sortie
  std::string sismo_file_;
  std::string fft_file_;
  std::string stats_mode_;
  // TP3 in-situ processing modes ("none" = disabled), set from the CLI
  std::string insitu_viz_;
  std::string insitu_compress_;
  std::string insitu_extra_;
  // Node traversal order used before compression ("natural" or "scan").
  // node_order_[k] = index of the node stored at position k; it is built once,
  // on the first compressed snapshot, from the node coordinates.
  std::string compress_order_ = "natural";
  std::vector<int> node_order_;
    // True: the CSV snapshots are not written (replaced by the .cmp files)
  bool compress_no_csv_ = false;
  // Steering (--insitu-extra steering): control file re-read every
  // steer_interval_ steps. See steerFromFile().
  std::string steer_file_;
  int steer_interval_ = 10;
  int steer_last_now_ = 0;        // last seen value of snapshot_now
  bool steer_stop_ = false;       // clean stop requested
  int slice_axis_ = -1;           // -1: full snapshot, else 0/1/2 = x/y/z
  double slice_value_ = 0.0;      // position of the slice plane
  double steer_total_ms_ = 0.0;   // time spent reading the control file
  long steer_reads_ = 0;          // number of reads
  // Compress the newest pressure field (column i1) with the pipeline given by
  // --insitu-compress (quantize, then RLE if requested) and log size/time in
  // the compression CSV. Called at snapshot steps from runInSituProcessing().
  void compressSnapshot(int step);
    // Compress each receiver trace with the --insitu-compress pipeline and print
  // the ratio and the reconstruction error (one line per receiver).
  void compressSeismograms();
  // Returns false to request a clean stop of the simulation.
  bool runInSituProcessing(int step);
  // Read the steering control file and apply its commands (snapshot interval,
  // slice plane, immediate snapshot, clean stop). Timed: see steer_total_ms_.
  void steerFromFile(int step);
  std::string visu_plane_;
  float visu_position_;
  int visu_width_;
  int visu_height_;
  std::string visu_format_;
  int visu_interval_;
  std::string visu_folder_;
  float visu_scale_;
  bool visu_fsync_ = false;
  std::string visu_backend_, visu_normalization_;
  int visu_window_ = 8;
  std::string maps_folder_;
  bool maps_images_ = true;
  std::unique_ptr<TP3Maps> maps_;

  std::unique_ptr<TP3Slice> visu_slice_;
  void savePPMSlice(int step, int column);
  int stats_interval_ = 0;  // intervalle des statistiques in situ
  double stats_total_ms_ = 0.0;
  int stats_count_ = 0;
  Kokkos::View<StreamStats*, Kokkos::DefaultExecutionSpace::memory_space> receiver_stats_device_;
  Kokkos::View<StreamStats, Kokkos::DefaultExecutionSpace::memory_space> pressure_stats_device_;
  Kokkos::View<StreamStats, Kokkos::HostSpace> pressure_stats_host_;
  Kokkos::View<float*, Kokkos::HostSpace> stats_host_column_;
  std::string sismo_stats_file_;
  std::string stats_file_;
  bool receiver_stats_enabled_ = false, keep_receiver_traces_ = true;
  int slice_interval_ = 0;
  std::string slice_plane_, slice_folder_;
  float slice_requested_ = 0, slice_coordinate_ = 0;
  vectorInt slice_nodes_;
  Kokkos::View<float*, Kokkos::DefaultExecutionSpace::memory_space> slice_values_;
  Kokkos::View<float*, Kokkos::HostSpace> slice_host_values_;
  bool slice_coords_written_ = false;
  void initSlice();
  void saveReceiverStatistics();
  std::ofstream stats_out_;
  float domain_size_[3] = {0};
  // snapshots
  bool is_snapshots_;
  int snap_time_interval_;
  std::string snap_folder_;
  std::string snap_format_, sismo_format_, metrics_file_;
  bool durable_outputs_ = false, record_receivers_ = true, quiet_ = false;
  bool coords_written_ = false;
  tp1::Metrics metrics_;
  arrayInt receiver_nodes_;
  void saveSeismograms();
  // physics
  bool isElastic_;
  // time parameters
  float dt_;
  float timemax_;
  int num_sample_;
  // source parameters
  const int myNumberOfRHS = 1;
  const float f0 = 5.;
  const int sourceOrder = 2;
  int myElementSource = 0;
  std::shared_ptr<model::ModelApi<float, int>> m_mesh;
  std::unique_ptr<SEMSolverBase> m_solver;
  SolverUtils myUtils;
  // arrays
  arrayReal myRHSTerm;
  arrayReal pnGlobal;
  vectorInt rhsElement;
  vectorInt rhsElementRcv;
  arrayReal rhsWeights;
  arrayReal rhsWeightsRcv;
  arrayReal pnAtReceiver;
  //sismos: lit le fichier des recepteurs
  //et renvois la liste des positions
  static std::vector<std::array<float, 3>> readReceivers(
     const std::string& filename);
  // initialize source and RHS
  void init_source();
  // allocate arrays and vectors
  void init_arrays();
    // statistiques in situ, version CPU (une seule passe, algorithme de Welford)
  void computePressureStatisticsCPU(int column, int step);
   // In-situ statistics, host version: explicit GPU->host copy of one column,
  // then a single-pass CPU loop. Copy and compute are timed separately.
  void computePressureStatisticsHostCopy(int column, int step);
  // Accumulate the in-situ stats time and write one CSV row (all modes)
  void recordStats(int step, const char* mode, double minP, double maxP,
                   double mean, double variance, double copyMs,
                   double computeMs, double totalMs, std::size_t bytes);
  // private methods to pars argv options
  int getPhysic(string physicArg);
  SolverFactory::implemType getImplem(string implemArg);
  SolverFactory::methodType getMethod(string methodArg);
  SolverFactory::meshType getMesh(string meshArg);
};
#endif /* SEMPROXY_HPP_ */
