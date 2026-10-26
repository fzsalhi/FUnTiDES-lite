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
#include "sem_proxy_options.h"

/* Streaming statistics (Welford's algorithm): constant memory*/

struct StreamStats
{
  long n = 0;
  double mean = 0.0;
  double m2 = 0.0;  // sum of squared deviations from the running mean
  double minVal = std::numeric_limits<double>::max();
  double maxVal = std::numeric_limits<double>::lowest();

  void add(double x)
  {
    n++;
    if (x < minVal) minVal = x;
    if (x > maxVal) maxVal = x;
    const double delta = x - mean;
    mean += delta / n;
    m2 += delta * (x - mean);
  }
  double variance() const { return n > 0 ? m2 / n : 0.0; }
};


/**
 * @class SEMproxy
 */
class SEMproxy
{
 public:
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
  // In-situ statistics of one pressure column computed on the device with
  // Kokkos::parallel_reduce. It must be public: nvcc forbids extended
  // lambdas (KOKKOS_LAMBDA) inside private/protected member functions.
  void computePressureStatistics(int column, int step);
 private:
  int i1 = 0;
  int i2 = 1;
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
  std::string stats_mode_;
  int stats_interval_ = 0;  // intervalle des statistiques in situ
  double stats_total_ms_ = 0.0;
  int stats_count_ = 0;
  std::vector<StreamStats> sismo_stats_;  // one entry per receiver
  std::ofstream stats_out_;
  float domain_size_[3] = {0};
  // snapshots
  bool is_snapshots_;
  int snap_time_interval_;
  std::string snap_folder_;
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
