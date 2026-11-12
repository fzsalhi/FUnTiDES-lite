#include <Kokkos_Core.hpp>
#include <chrono>
//************************************************************************
//   proxy application v.0.0.1
//
//  semproxy.cpp: the main interface of  proxy application
//
//************************************************************************
#include "sem_proxy.h"
#include <cartesian_struct_builder.h>
#include <cartesian_unstruct_builder.h>
#include <sem_solver_acoustic.h>
#include <source_and_receiver_utils.h>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <variant>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <limits>
#include "compression_gpu.hpp"
#include "compression.hpp"
#include "rle.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace tp1 {
std::string jsonString(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << static_cast<int>(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

std::uintmax_t finish(std::ofstream& stream, const std::filesystem::path& path,
                     bool durable, double& fsync_s) {
  stream.close();
  if (!stream) throw std::runtime_error("Output failed: " + path.string());
  if (durable) {
    const auto start = Clock::now();
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), path.string());
    const int result = ::fsync(fd);
    const int saved_errno = errno;
    const int close_result = ::close(fd);
    if (result != 0) throw std::system_error(saved_errno, std::generic_category(), "fsync");
    if (close_result != 0) throw std::system_error(errno, std::generic_category(), "close");
    fsync_s += seconds(start);
  }
  return std::filesystem::file_size(path);
}

void writeFloat32(std::ofstream& stream, const std::vector<float>& values) {
  static_assert(sizeof(float) == 4, "TP1 binary format requires IEEE float32");
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian) == 1) {
    stream.write(reinterpret_cast<const char*>(values.data()), values.size() * 4);
  } else {
    for (float value : values) {
      char bytes[4];
      std::memcpy(bytes, &value, 4);
      std::reverse(bytes, bytes + 4);
      stream.write(bytes, 4);
    }
  }
}
}  // namespace tp1

using namespace SourceAndReceiverUtils;
SEMproxy::SEMproxy(const SemProxyOptions& opt)
{
  const int order = opt.order;
  nb_elements_[0] = opt.ex;
  nb_elements_[1] = opt.ey;
  nb_elements_[2] = opt.ez;
  nb_nodes_[0] = opt.ex * order + 1;
  nb_nodes_[1] = opt.ey * order + 1;
  nb_nodes_[2] = opt.ez * order + 1;
  const float spongex = opt.boundaries_size;
  const float spongey = opt.boundaries_size;
  const float spongez = opt.boundaries_size;
  const std::array<float, 3> sponge_size = {spongex, spongey, spongez};
  src_coord_[0] = opt.srcx;
  src_coord_[1] = opt.srcy;
  src_coord_[2] = opt.srcz;
  domain_size_[0] = opt.lx;
  domain_size_[1] = opt.ly;
  domain_size_[2] = opt.lz;
  rcv_coord_[0] = opt.rcvx;
  rcv_coord_[1] = opt.rcvy;
  rcv_coord_[2] = opt.rcvz;
    // sismos : liste des recepteurs
  if (opt.rcv_file.empty())
    rcv_coords_.push_back(rcv_coord_);  // pas de fichier : ancien recepteur unique
  else
    rcv_coords_ = readReceivers(opt.rcv_file);
  for (const auto& receiver : rcv_coords_)
    for (int d = 0; d < 3; ++d)
      if (!std::isfinite(receiver[d]) || receiver[d] < 0 || receiver[d] > domain_size_[d])
        throw std::runtime_error("Receiver coordinates must be finite and inside the domain");
  sismo_file_ = opt.sismo_file;
  fft_file_ = opt.fft_file;
#ifndef SEMPROXY_ENABLE_CUFFT
  if (!fft_file_.empty())
    throw std::runtime_error("--fft-file requires a build with SEMPROXY_ENABLE_CUFFT=ON");
#endif
  stats_interval_ = opt.stats_interval;
  stats_file_ = opt.stats_file;
  sismo_stats_file_ = opt.sismo_stats_file;
  receiver_stats_enabled_ = opt.record_receivers && (stats_interval_ > 0 || !sismo_stats_file_.empty());
  keep_receiver_traces_ = !opt.sismo_file.empty() || !opt.fft_file.empty() || opt.insitu_compress != "none";
  slice_interval_ = opt.slice_interval;
  slice_plane_ = opt.slice_plane;
  slice_folder_ = opt.slice_folder;
  slice_requested_ = opt.slice_position;
    if (!opt.stats_file.empty())
  {
    const std::filesystem::path statsPath(opt.stats_file);
    if (!statsPath.parent_path().empty()) std::filesystem::create_directories(statsPath.parent_path());
    stats_out_.open(opt.stats_file);
    if (!stats_out_)
      throw std::runtime_error("Cannot open stats file: " + opt.stats_file);
    stats_out_ << std::setprecision(9);
    stats_out_ << "step,min,max,mean,variance,time_ms,mode,copy_ms,compute_ms,bytes,time_s,count\n";
  }
  
  stats_mode_ = opt.stats_mode;
  if (stats_mode_ != "cpu" && stats_mode_ != "gpu" && stats_mode_ != "hostcopy")
    throw std::runtime_error("Unknown --stats-mode: " + stats_mode_);
  // TP3 in-situ processing modes (each owner validates its own values later)
  insitu_viz_ = opt.insitu_viz;
  insitu_compress_ = opt.insitu_compress;
  std::string err;
  if (!parse_compress(insitu_compress_, compress_cfg_, err))
    throw std::runtime_error("--insitu-compress: " + err);
  insitu_extra_ = opt.insitu_extra;


  // Steering options
  steer_file_ = opt.steer_file;
  steer_interval_ = opt.steer_interval;
  if (steer_interval_ < 1)
    throw std::runtime_error("--steer-interval must be >= 1");
  if (opt.insitu_extra == "steering" && steer_file_.empty())
    throw std::runtime_error("--insitu-extra steering requires --steer-file");

     compress_no_csv_ = (opt.compress_no_csv != 0);
  if (compress_no_csv_ && opt.insitu_compress == "none")
    throw std::runtime_error("--compress-no-csv requires --insitu-compress");
  // Node traversal order used before compression
  compress_order_ = opt.compress_order;
  if (compress_order_ != "natural" && compress_order_ != "scan" && compress_order_ != "scanz")
    throw std::runtime_error(
        "--compress-order must be 'natural' or 'scan' or 'scanz', got '" +
        compress_order_ + "'");
  // Parse --insitu-compress (e.g. "raw+rle", "int8+delta+rle"). The syntax was
  // already checked in SemProxyOptions::validate(); here we keep the result.
  {
    std::string err;
    if (!parse_compress(insitu_compress_, compress_cfg_, err))
      throw std::runtime_error(err);
  }
   // CSV of the compression ratios of compressSnapshot(), only when a file name
  // is given and a lossless stage (RLE / DELTA) is requested. The other
  // configurations go through the quantization block, which writes its own CSV.
  if (compress_cfg_.enabled && (compress_cfg_.rle || compress_cfg_.delta) &&
      !opt.compress_file.empty())
  {
    compress_out_.open(opt.compress_file);
    if (!compress_out_)
      throw std::runtime_error("Cannot open compress file: " +
                               opt.compress_file);
       compress_out_ << "step,raw_bytes,stored_bytes,ratio,time_ms,write_ms,"
                     "csv_bytes,rmse_norm,max_abs_norm,d_min,d_max,d_mean,"
                     "d_var_rel,rcv_orig,rcv_rec\n";
  }
  visu_plane_ = opt.visu_plane;
  visu_position_ = opt.visu_position;
  visu_width_ = opt.visu_width;
  visu_height_ = opt.visu_height;
  visu_format_ = opt.visu_format;
  visu_interval_ = opt.visu_interval;
  visu_folder_ = opt.visu_folder;
  visu_scale_ = opt.visu_scale;
  visu_fsync_ = opt.visu_fsync || opt.io_sync == "fsync";
  visu_backend_ = opt.visu_backend;
  visu_normalization_ = opt.visu_normalization;
  visu_window_ = opt.visu_window;
  maps_folder_ = opt.maps_folder;
  maps_images_ = opt.maps_images;

  if (insitu_viz_ != "none" && insitu_viz_ != "ppm")
      throw std::runtime_error("Unknown visualization mode: " + insitu_viz_);
  // one streaming accumulator per receiver (rcv_coords_ is already filled here)
  if (receiver_stats_enabled_) {
    receiver_stats_device_ = decltype(receiver_stats_device_)("receiverStats", rcv_coords_.size());
    Kokkos::deep_copy(receiver_stats_device_, StreamStats{});
  }
  if (stats_interval_ > 0) {
    pressure_stats_device_ = decltype(pressure_stats_device_)("pressureStats");
    pressure_stats_host_ = decltype(pressure_stats_host_)("pressureStatsHost");
  }

  std::cout << "Number of receivers is " << rcv_coords_.size() << std::endl;
  bool isModelOnNodes = opt.isModelOnNodes;
  isElastic_ = opt.isElastic;
  is_snapshots_ = opt.is_snapshots;
  snap_time_interval_ = opt.snap_time_interval;
  snap_folder_ = opt.snap_folder;
  snap_format_ = opt.snap_format;
  sismo_format_ = opt.sismo_format;
  metrics_file_ = opt.metrics_file;
  durable_outputs_ = opt.io_sync == "fsync";
  record_receivers_ = opt.record_receivers;
  quiet_ = opt.quiet;
  cout << boolalpha;
  bool isElastic = isElastic_;
  const SolverFactory::methodType methodType = getMethod(opt.method);
  const SolverFactory::implemType implemType = getImplem(opt.implem);
  const SolverFactory::meshType meshType = getMesh(opt.mesh);
  const SolverFactory::modelLocationType modelLocation =
      isModelOnNodes ? SolverFactory::modelLocationType::OnNodes
                     : SolverFactory::modelLocationType::OnElements;
  const SolverFactory::physicType physicType = SolverFactory::physicType::Acoustic;
  float lx = domain_size_[0];
  float ly = domain_size_[1];
  float lz = domain_size_[2];
  int ex = nb_elements_[0];
  int ey = nb_elements_[1];
  int ez = nb_elements_[2];
  if (meshType == SolverFactory::Struct)
  {
    switch (order)
    {
      case 1: {
        model::CartesianStructBuilder<float, int, 1> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      case 2: {
        model::CartesianStructBuilder<float, int, 2> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      case 3: {
        model::CartesianStructBuilder<float, int, 3> builder(
            ex, lx, ey, ly, ez, lz, isModelOnNodes);
        m_mesh = builder.getModel();
        break;
      }
      default:
        throw std::runtime_error(
            "Order other than 1 2 3 is not supported (semproxy)");
    }
  }
  else if (meshType == SolverFactory::Unstruct)
  {
    model::CartesianParams<float, int> param(order, ex, ey, ez, lx, ly, lz,
                                             isModelOnNodes);
    model::CartesianUnstructBuilder<float, int> builder(param);
    m_mesh = builder.getModel();
  }
  else
  {
    throw std::runtime_error("Incorrect mesh type (SEMproxy ctor.)");
  }
  if (insitu_viz_ == "ppm" || (opt.insitu_maps != "none" && opt.maps_images))
  {
      const auto setupStart=tp1::Clock::now();
      if (insitu_viz_ == "ppm") std::filesystem::create_directories(visu_folder_);

      visu_slice_ = std::make_unique<TP3Slice>(
          *m_mesh,
          nb_elements_,
          domain_size_,
          visu_plane_,
          visu_position_,
          visu_width_,
          visu_height_, visu_backend_, visu_normalization_, visu_window_
      );
      Kokkos::fence();
      metrics_.ppm_setup_s=tp1::seconds(setupStart);
  }
  if (opt.insitu_maps != "none")
      maps_=std::make_unique<TP3Maps>(m_mesh->getNumberOfNodes(),opt.insitu_maps,opt.maps_threshold,opt.maps_reference_interval);
  // time parameters
  if (opt.autodt)
  {
    float cfl_factor = (order == 2) ? 0.5 : 0.7;
    dt_ = find_cfl_dt(cfl_factor);
  }
  else
  {
    dt_ = opt.dt;
  }
  timemax_ = opt.timemax;
  if (!std::isfinite(dt_) || dt_ <= 0 || double(timemax_) / dt_ > std::numeric_limits<int>::max())
    throw std::runtime_error("Invalid time step or too many time steps");
  num_sample_ = timemax_ / dt_;
  if (num_sample_ < 1)
    throw std::runtime_error("timemax must cover at least one time step");
  m_solver = SolverFactory::createSolver(methodType, implemType, meshType,
                                         modelLocation, physicType, order);
  m_solver->computeFEInit(*m_mesh, sponge_size, opt.surface_sponge,
                          opt.taper_delta);
  initFiniteElem();
  if (stats_interval_ > 0 && stats_mode_ == "hostcopy")
    stats_host_column_ = decltype(stats_host_column_)("statsHostColumn", m_mesh->getNumberOfNodes());
  if (slice_interval_ > 0) initSlice();
  // Precompute the interpolation stencil on the host. Sampling itself runs in
  // the execution space so CUDA does not migrate the field to the CPU at every step.
  const int points = m_mesh->getNumberOfPointsPerElement();
  receiver_nodes_ = allocateArray2D<arrayInt>(rcv_coords_.size(), points, "receiverNodes");
  for (std::size_t r = 0; r < rcv_coords_.size(); ++r)
    for (int k = 0; k <= order; ++k)
      for (int j = 0; j <= order; ++j)
        for (int i = 0; i <= order; ++i) {
          const int q = i + j * (order + 1) + k * (order + 1) * (order + 1);
          receiver_nodes_(r, q) = m_mesh->globalNodeIndex(rhsElementRcv[r], i, j, k);
        }
  Kokkos::fence();
  std::cout << "Number of node is " << m_mesh->getNumberOfNodes() << std::endl;
  std::cout << "Number of element is " << m_mesh->getNumberOfElements()
            << std::endl;
  std::cout << "Launching the Method " << opt.method << ", the implementation "
            << opt.implem << " and the mesh is " << opt.mesh << std::endl;
  std::cout << "Model is on " << (isModelOnNodes ? "nodes" : "elements")
            << std::endl;
  std::cout << "Physics type is " << (isElastic ? "elastic" : "acoustic")
            << std::endl;
  std::cout << "Order of approximation will be " << order << std::endl;
  std::cout << "Time step is " << dt_ << "s" << std::endl;
  std::cout << "Simulated time is " << timemax_ << "s" << std::endl;
}
void SEMproxy::sampleReceivers(int timestep)
{
  auto pressure = pnGlobal;
  auto nodes = receiver_nodes_;
  auto weights = rhsWeightsRcv;
  auto traces = pnAtReceiver;
  const int column = i1;  // updatePressureField writes the new solution here
  const int target = keep_receiver_traces_ ? timestep : 0;
  const int points = m_mesh->getNumberOfPointsPerElement();
  Kokkos::parallel_for("sample_receivers", rcv_coords_.size(), KOKKOS_LAMBDA(int r) {
    float value = 0;
    for (int q = 0; q < points; ++q)
      value += pressure(nodes(r, q), column) * weights(r, q);
    traces(r, target) = value;
  });
}

void SEMproxy::run()
{
  Kokkos::fence();
  const auto runStart = tp1::Clock::now();
  SEMsolverDataAcoustic solverData(i1, i2, myRHSTerm, pnGlobal, rhsElement, rhsWeights);
  for (int step = 0; step < num_sample_; ++step)
  {
    const auto computeStart = tp1::Clock::now();
    m_solver->computeOneStep(dt_, step, solverData);
    Kokkos::fence();  // account for device work in the compute stage
    metrics_.compute_s += tp1::seconds(computeStart);
    metrics_.completed_steps = step + 1;

    if (record_receivers_) {
      const auto sampleStart = tp1::Clock::now();
      sampleReceivers(step);
      Kokkos::fence();
      metrics_.receiver_sample_s += tp1::seconds(sampleStart);
    }
    const auto otherStart = tp1::Clock::now();
    if (!quiet_ && step % 50 == 0)
      m_solver->outputSolutionValues(step, i1, rhsElement[0], pnGlobal, "pnGlobal");
    if (receiver_stats_enabled_) {
      const auto statsStart = tp1::Clock::now();
      updateReceiverStatistics(step);
      Kokkos::fence();
      metrics_.sismo_stats_s += tp1::seconds(statsStart);
    }
    metrics_.other_processing_s += tp1::seconds(otherStart);

    if (is_snapshots_ && step % snap_time_interval_ == 0 && !compress_no_csv_)
      saveSnapshot(step);

    if (slice_interval_ > 0 && step % slice_interval_ == 0) saveSlice(step);
    const auto processingStart = tp1::Clock::now();
    if (stats_interval_ > 0 && step % stats_interval_ == 0) {
      if (stats_mode_ == "gpu") computePressureStatistics(i1, step);
      else if (stats_mode_ == "hostcopy") computePressureStatisticsHostCopy(i1, step);
      else computePressureStatisticsCPU(i1, step);
    }
    const bool keepRunning = runInSituProcessing(step);
    Kokkos::fence();
    metrics_.other_processing_s += tp1::seconds(processingStart);
    if (!keepRunning) break;
    swap(i1, i2);
    std::swap(solverData.m_i1, solverData.m_i2);
  }
#ifdef SEMPROXY_ENABLE_CUFFT
  if (!fft_file_.empty()) {
    metrics_.fft = computeSismoFFTcuFFT(pnAtReceiver, rcv_coords_.size(),
        metrics_.completed_steps, dt_, fft_file_, durable_outputs_);
    metrics_.fsync_s += metrics_.fft.fsync_s;
  }
#endif
  if (maps_) {
    std::vector<float> coordinates(std::size_t(m_mesh->getNumberOfNodes())*3);
    for(int n=0;n<m_mesh->getNumberOfNodes();++n)
      for(int d=0;d<3;++d) coordinates[3*n+d]=m_mesh->nodeCoord(n,d);
    maps_->save(maps_folder_,coordinates,metrics_.completed_steps,dt_,durable_outputs_,maps_images_?visu_slice_.get():nullptr);
    metrics_.fsync_s+=maps_->metrics().fsync_s;
  }
  if (insitu_compress_ != "none") compressSeismograms();
  if (insitu_extra_ == "steering")
    cout << "---- Steering : " << steer_reads_ << " reads, total "
         << steer_total_ms_ << " ms." << endl;
  if (receiver_stats_enabled_) saveReceiverStatistics();
  if (stats_out_.is_open()) {
    const auto statsWrite = tp1::Clock::now();
    metrics_.stats_bytes = tp1::finish(stats_out_, stats_file_, durable_outputs_, metrics_.fsync_s);
    metrics_.stats_write_s += tp1::seconds(statsWrite);
  }
  cout << "---- Elapsed Stats Time : " << stats_total_ms_ / 1E3
       << " seconds (" << stats_count_ << " calls)." << endl;
  if (!sismo_file_.empty()) saveSeismograms();
  Kokkos::fence();
  metrics_.run_s = tp1::seconds(runStart);
  cout << "---- Elapsed Kernel Time : " << metrics_.compute_s << " seconds." << endl;
  cout << "---- Elapsed Output Time : "
       << metrics_.receiver_sample_s + metrics_.snapshot_write_s + metrics_.other_processing_s
       << " seconds." << endl;
  cout << "---- Elapsed Save Time : " << metrics_.snapshot_write_s + metrics_.sismo_write_s
       << " seconds." << endl;
}

// Initialize arrays
void SEMproxy::init_arrays()
{
  cout << "Allocate host memory for source and pressure values ..." << endl;
  rhsElement = allocateVector<vectorInt>(myNumberOfRHS, "rhsElement");
  rhsWeights = allocateArray2D<arrayReal>(
      myNumberOfRHS, m_mesh->getNumberOfPointsPerElement(), "RHSWeight");
  myRHSTerm = allocateArray2D<arrayReal>(myNumberOfRHS, num_sample_, "RHSTerm");
  pnGlobal =
      allocateArray2D<arrayReal>(m_mesh->getNumberOfNodes(), 2, "pnGlobal");
  const int nRcv = rcv_coords_.size();
  pnAtReceiver = allocateArray2D<arrayReal>(nRcv, keep_receiver_traces_ ? num_sample_ : 1, "pnAtReceiver");
  // Receiver
  rhsElementRcv = allocateVector<vectorInt>(nRcv, "rhsElementRcv");
  rhsWeightsRcv = allocateArray2D<arrayReal>(
      nRcv, m_mesh->getNumberOfPointsPerElement(), "RHSWeightRcv");
}
// Initialize sources
void SEMproxy::init_source()
{
  arrayReal myRHSLocation = allocateArray2D<arrayReal>(1, 3, "RHSLocation");
  // std::cout << "All source are currently are coded on element 50." <<
  // std::endl;
  std::cout << "All source are currently are coded on middle element."
            << std::endl;
  int ex = nb_elements_[0];
  int ey = nb_elements_[1];
  int ez = nb_elements_[2];
  float lx = domain_size_[0];
  float ly = domain_size_[1];
  float lz = domain_size_[2];
  // Get source element index
  int source_index = floor((src_coord_[0] * ex) / lx) +
                     floor((src_coord_[1] * ey) / ly) * ex +
                     floor((src_coord_[2] * ez) / lz) * ey * ex;
  for (int i = 0; i < 1; i++)
  {
    rhsElement[i] = source_index;
  }
  // Get coordinates of the corners of the sourc element
  float cornerCoords[8][3];
  int I = 0;
  int nodes_corner[2] = {0, m_mesh->getOrder()};
  for (int k : nodes_corner)
  {
    for (int j : nodes_corner)
    {
      for (int i : nodes_corner)
      {
        int nodeIdx = m_mesh->globalNodeIndex(rhsElement[0], i, j, k);
        cornerCoords[I][0] = m_mesh->nodeCoord(nodeIdx, 0);
        cornerCoords[I][2] = m_mesh->nodeCoord(nodeIdx, 2);
        cornerCoords[I][1] = m_mesh->nodeCoord(nodeIdx, 1);
        I++;
      }
    }
  }
  // initialize source term
  vector<float> sourceTerm =
      myUtils.computeSourceTerm(num_sample_, dt_, f0, sourceOrder);
  for (int j = 0; j < num_sample_; j++)
  {
    myRHSTerm(0, j) = sourceTerm[j];
    if (j % 100 == 0)
      cout << "Sample " << j << "\t: sourceTerm = " << sourceTerm[j] << endl;
  }
  // get element number of source term
  myElementSource = rhsElement[0];
  cout << "Element number for the source location: " << myElementSource << endl
       << endl;
  int order = m_mesh->getOrder();
  switch (order)
  {
    case 1:
      SourceAndReceiverUtils::ComputeRHSWeights<1>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    case 2:
      SourceAndReceiverUtils::ComputeRHSWeights<2>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    case 3:
      SourceAndReceiverUtils::ComputeRHSWeights<3>(cornerCoords, src_coord_,
                                                   rhsWeights);
      break;
    default:
      throw std::runtime_error("Unsupported order: " + std::to_string(order));
  }
  // Receivers computation : meme travail qu'avant, mais pour CHAQUE recepteur
  const int nPts = m_mesh->getNumberOfPointsPerElement();
  // ComputeRHSWeights ecrit toujours dans la ligne 0 du tableau qu'on lui
  // donne : on lui donne un tableau temporaire d'1 ligne, puis on recopie
  arrayReal tmpWeights = allocateArray2D<arrayReal>(1, nPts, "tmpRcvWeights");
  for (int r = 0; r < (int)rcv_coords_.size(); r++)
  {
    // 1. maille qui contient le recepteur r
    int receiver_index = std::min(ex - 1, int(floor((rcv_coords_[r][0] * ex) / lx))) +
                         std::min(ey - 1, int(floor((rcv_coords_[r][1] * ey) / ly))) * ex +
                         std::min(ez - 1, int(floor((rcv_coords_[r][2] * ez) / lz))) * ey * ex;
    rhsElementRcv[r] = receiver_index;
    cout << "Receiver " << r << " is in element " << receiver_index << endl;
    // 2. coordonnees des 8 coins de cette maille
    float cornerCoordsRcv[8][3];
    I = 0;
    for (int k : nodes_corner)
    {
      for (int j : nodes_corner)
      {
        for (int i : nodes_corner)
        {
          int nodeIdx = m_mesh->globalNodeIndex(rhsElementRcv[r], i, j, k);
          cornerCoordsRcv[I][0] = m_mesh->nodeCoord(nodeIdx, 0);
          cornerCoordsRcv[I][2] = m_mesh->nodeCoord(nodeIdx, 2);
          cornerCoordsRcv[I][1] = m_mesh->nodeCoord(nodeIdx, 1);
          I++;
        }
      }
    }
    // 3. poids d'interpolation (dans le tableau temporaire)
    switch (order)
    {
      case 1:
        SourceAndReceiverUtils::ComputeRHSWeights<1>(
            cornerCoordsRcv, rcv_coords_[r], tmpWeights);
        break;
      case 2:
        SourceAndReceiverUtils::ComputeRHSWeights<2>(
            cornerCoordsRcv, rcv_coords_[r], tmpWeights);
        break;
      case 3:
        SourceAndReceiverUtils::ComputeRHSWeights<3>(
            cornerCoordsRcv, rcv_coords_[r], tmpWeights);
        break;
      default:
        throw std::runtime_error("Unsupported order: " +
                                 std::to_string(order));
    }
    // 4. on range les poids dans la ligne r du vrai tableau
    for (int i = 0; i < nPts; i++) rhsWeightsRcv(r, i) = tmpWeights(0, i);
  }
}
SolverFactory::implemType SEMproxy::getImplem(string implemArg)
{
  if (implemArg == "makutu") return SolverFactory::MAKUTU;
  if (implemArg == "shiva") return SolverFactory::SHIVA;
  throw std::invalid_argument(
      "Implentation type does not follow any valid type.");
}
SolverFactory::meshType SEMproxy::getMesh(string meshArg)
{
  if (meshArg == "cartesian") return SolverFactory::Struct;
  if (meshArg == "ucartesian") return SolverFactory::Unstruct;
  std::cout << "Mesh type found is " << meshArg << std::endl;
  throw std::invalid_argument("Mesh type does not follow any valid type.");
}
SolverFactory::methodType SEMproxy::getMethod(string methodArg)
{
  if (methodArg == "sem") return SolverFactory::SEM;
  if (methodArg == "dg") return SolverFactory::DG;
  throw std::invalid_argument("Method type does not follow any valid type.");
}
float SEMproxy::find_cfl_dt(float cfl_factor)
{
  float sqrtDim3 = 1.73;  // to change for 2d
  float min_spacing = m_mesh->getMinSpacing();
  float v_max = m_mesh->getMaxSpeed();
  float dt = cfl_factor * min_spacing / (sqrtDim3 * v_max);
  return dt;
}
namespace fs = std::filesystem;

void SEMproxy::saveSnapshot(int timestep)
{
  Kokkos::fence();
  const auto start = tp1::Clock::now();
  fs::create_directories(snap_folder_);
  const int count = m_mesh->getNumberOfNodes();
  std::vector<int> nodes;
  nodes.reserve(count);
  double distance = 0;
  if (slice_axis_ >= 0) {
    distance = std::numeric_limits<double>::max();
    for (int n = 0; n < count; ++n)
      distance = std::min(distance, std::abs(m_mesh->nodeCoord(n, slice_axis_) - slice_value_));
    distance += 1e-5 * (1 + std::abs(slice_value_));
  }
  for (int n = 0; n < count; ++n)
    if (slice_axis_ < 0 || std::abs(m_mesh->nodeCoord(n, slice_axis_) - slice_value_) <= distance)
      nodes.push_back(n);

  std::ostringstream stem;
  stem << "snapshot_it_" << std::setw(6) << std::setfill('0') << timestep;
  const fs::path base = fs::path(snap_folder_) / stem.str();
  auto finish = [&](std::ofstream& out, const fs::path& path) {
    metrics_.snapshot_bytes += tp1::finish(out, path, durable_outputs_, metrics_.fsync_s);
  };
  std::string coordsName;
  if (snap_format_ != "csv") {
    coordsName = slice_axis_ >= 0 ? "coords_" + stem.str() + ".bin" : "coords.bin";
    if (slice_axis_ >= 0 || !coords_written_) {
      std::vector<float> coords(nodes.size() * 3);
      for (std::size_t k = 0; k < nodes.size(); ++k)
        for (int d = 0; d < 3; ++d) coords[k * 3 + d] = m_mesh->nodeCoord(nodes[k], d);
      const auto path = fs::path(snap_folder_) / coordsName;
      std::ofstream out(path, std::ios::binary);
      tp1::writeFloat32(out, coords);
      finish(out, path);
      if (slice_axis_ < 0) coords_written_ = true;
    }
    std::vector<float> pressure(nodes.size());
    for (std::size_t k = 0; k < nodes.size(); ++k) pressure[k] = pnGlobal(nodes[k], i1);
    const auto path = base.string() + ".bin";
    std::ofstream out(path, std::ios::binary);
    tp1::writeFloat32(out, pressure);
    finish(out, path);
  }
  if (snap_format_ != "bin") {
    const auto path = base.string() + ".csv";
    std::ofstream out(path);
    out << "x,y,z,pressure\n" << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (int n : nodes)
      out << m_mesh->nodeCoord(n, 0) << ',' << m_mesh->nodeCoord(n, 1) << ','
          << m_mesh->nodeCoord(n, 2) << ',' << pnGlobal(n, i1) << '\n';
    finish(out, path);
  }
  const auto path = base.string() + ".json";
  std::ofstream meta(path);
  meta << std::setprecision(17)
       << "{\"schema\":\"funtides.snapshot.v1\",\"dtype\":\"<f4\",\"count\":" << nodes.size()
       << ",\"step\":" << timestep << ",\"time_s\":" << (timestep + 1) * double(dt_)
       << ",\"dt_s\":" << dt_ << ",\"pressure_column\":\"new\",\"coordinates\":"
       << tp1::jsonString(coordsName) << "}\n";
  finish(meta, path);
  ++metrics_.snapshot_count;
  metrics_.snapshot_write_s += tp1::seconds(start);
}

void SEMproxy::saveSeismograms()
{
  Kokkos::fence();
  const auto start = tp1::Clock::now();
  fs::path base(sismo_file_);
  if (!base.parent_path().empty()) fs::create_directories(base.parent_path());
  auto finish = [&](std::ofstream& out, const fs::path& path) {
    metrics_.sismo_bytes += tp1::finish(out, path, durable_outputs_, metrics_.fsync_s);
  };
  if (sismo_format_ != "bin") {
    auto path = base;
    if (sismo_format_ == "both") path.replace_extension(".csv");
    std::ofstream out(path);
    out << "time";
    for (std::size_t r = 0; r < rcv_coords_.size(); ++r) out << ",receiver_" << r;
    out << '\n' << std::setprecision(17);
    for (int t = 0; t < metrics_.completed_steps; ++t) {
      out << (t + 1) * double(dt_);
      for (std::size_t r = 0; r < rcv_coords_.size(); ++r) out << ',' << pnAtReceiver(r, t);
      out << '\n';
    }
    finish(out, path);
  }
  if (sismo_format_ != "csv") {
    auto path = base;
    if (sismo_format_ == "both") path.replace_extension(".bin");
    std::ofstream out(path, std::ios::binary);
    // Sample-major blocks: no assumption about the LayoutLeft/Right of the Kokkos view.
    std::vector<float> block;
    for (int t = 0; t < metrics_.completed_steps;) {
      const int end = std::min(t + 1024, metrics_.completed_steps);
      block.clear();
      block.reserve((end - t) * rcv_coords_.size());
      for (; t < end; ++t)
        for (std::size_t r = 0; r < rcv_coords_.size(); ++r) block.push_back(pnAtReceiver(r, t));
      tp1::writeFloat32(out, block);
    }
    finish(out, path);
  }
  auto path = base;
  path.replace_extension(".json");
  std::ofstream meta(path);
  meta << std::setprecision(17)
       << "{\"schema\":\"funtides.seismograms.v1\",\"dtype\":\"<f4\",\"layout\":\"sample-major\","
       << "\"samples\":" << metrics_.completed_steps << ",\"dt_s\":" << dt_
       << ",\"time_start_s\":" << dt_ << ",\"receivers\":[";
  for (std::size_t r = 0; r < rcv_coords_.size(); ++r) {
    if (r) meta << ',';
    meta << '[' << rcv_coords_[r][0] << ',' << rcv_coords_[r][1] << ',' << rcv_coords_[r][2] << ']';
  }
  meta << "]}\n";
  finish(meta, path);
  metrics_.sismo_write_s += tp1::seconds(start);
}

void SEMproxy::writeMetrics(double initial_s, double total_s) const
{
  if (metrics_file_.empty()) return;
  const fs::path path(metrics_file_);
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  std::ofstream out(path);
  const std::string space = Kokkos::DefaultExecutionSpace::name();
  const bool gpu = space == "Cuda" || space == "HIP" || space == "SYCL";
  out << std::setprecision(17)
      << "{\n  \"schema\": \"funtides.metrics.v1\",\n"
      << "  \"backend\": " << tp1::jsonString(gpu ? "gpu" : "cpu")
      << ",\n  \"execution_space\": " << tp1::jsonString(space)
      << ",\n  \"compiler\": " << tp1::jsonString(__VERSION__)
      << ",\n  \"pressure_bytes_per_value\": 4,\n  \"nodes\": " << m_mesh->getNumberOfNodes()
      << ",\n  \"elements\": " << m_mesh->getNumberOfElements()
      << ",\n  \"order\": " << m_mesh->getOrder()
      << ",\n  \"steps\": " << metrics_.completed_steps
      << ",\n  \"receivers\": " << (record_receivers_ ? rcv_coords_.size() : 0)
      << ",\n  \"dt_s\": " << dt_ << ",\n  \"requested_time_s\": " << timemax_
      << ",\n  \"simulated_time_s\": " << metrics_.completed_steps * double(dt_)
      << ",\n  \"snapshot_format\": " << tp1::jsonString(snap_format_)
      << ",\n  \"sismo_format\": " << tp1::jsonString(sismo_format_)
      << ",\n  \"io_sync\": " << tp1::jsonString(durable_outputs_ ? "fsync" : "buffered")
      << ",\n  \"initial_s\": " << initial_s << ",\n  \"total_s\": " << total_s
      << ",\n  \"run_s\": " << metrics_.run_s
      << ",\n  \"compute_s\": " << metrics_.compute_s
      << ",\n  \"receiver_sample_s\": " << metrics_.receiver_sample_s
      << ",\n  \"snapshot_write_s\": " << metrics_.snapshot_write_s
      << ",\n  \"sismo_write_s\": " << metrics_.sismo_write_s
      << ",\n  \"other_processing_s\": " << metrics_.other_processing_s
      << ",\n  \"fsync_s\": " << metrics_.fsync_s
      << ",\n  \"snapshot_count\": " << metrics_.snapshot_count
      << ",\n  \"snapshot_bytes\": " << metrics_.snapshot_bytes
      << ",\n  \"sismo_bytes\": " << metrics_.sismo_bytes
      << ",\n  \"stats_compute_s\": " << metrics_.stats_compute_s
      << ",\n  \"stats_copy_s\": " << metrics_.stats_copy_s
      << ",\n  \"stats_write_s\": " << metrics_.stats_write_s
      << ",\n  \"stats_bytes\": " << metrics_.stats_bytes
      << ",\n  \"stats_calls\": " << stats_count_
      << ",\n  \"stats_transfer_bytes\": " << metrics_.stats_transfer_bytes
      << ",\n  \"sismo_stats_s\": " << metrics_.sismo_stats_s
      << ",\n  \"sismo_stats_copy_s\": " << metrics_.sismo_stats_copy_s
      << ",\n  \"sismo_stats_write_s\": " << metrics_.sismo_stats_write_s
      << ",\n  \"sismo_stats_bytes\": " << metrics_.sismo_stats_bytes
      << ",\n  \"receiver_trace_bytes\": " << pnAtReceiver.extent(0) * pnAtReceiver.extent(1) * sizeof(float)
      << ",\n  \"receiver_stats_memory_bytes\": " << receiver_stats_device_.extent(0) * sizeof(StreamStats)
      << ",\n  \"slice_setup_s\": " << metrics_.slice_setup_s
      << ",\n  \"slice_extract_s\": " << metrics_.slice_extract_s
      << ",\n  \"slice_copy_s\": " << metrics_.slice_copy_s
      << ",\n  \"slice_write_s\": " << metrics_.slice_write_s
      << ",\n  \"slice_count\": " << metrics_.slice_count
      << ",\n  \"slice_nodes\": " << slice_nodes_.extent(0)
      << ",\n  \"slice_bytes\": " << metrics_.slice_bytes
      << ",\n  \"fft_prepare_s\": " << metrics_.fft.prepare_s
      << ",\n  \"fft_plan_s\": " << metrics_.fft.plan_s
      << ",\n  \"fft_h2d_s\": " << metrics_.fft.h2d_s
      << ",\n  \"fft_compute_s\": " << metrics_.fft.compute_s
      << ",\n  \"fft_d2h_s\": " << metrics_.fft.d2h_s
      << ",\n  \"fft_write_s\": " << metrics_.fft.write_s
      << ",\n  \"fft_total_s\": " << metrics_.fft.total_s
      << ",\n  \"fft_h2d_bytes\": " << metrics_.fft.h2d_bytes
      << ",\n  \"fft_d2h_bytes\": " << metrics_.fft.d2h_bytes
      << ",\n  \"fft_output_bytes\": " << metrics_.fft.output_bytes
      << ",\n  \"ppm_setup_s\": " << metrics_.ppm_setup_s
      << ",\n  \"ppm_s\": " << metrics_.ppm_s
      << ",\n  \"ppm_extract_s\": " << metrics_.ppm_extract_s
      << ",\n  \"ppm_color_s\": " << metrics_.ppm_color_s
      << ",\n  \"ppm_copy_s\": " << metrics_.ppm_copy_s
      << ",\n  \"ppm_encode_s\": " << metrics_.ppm_encode_s
      << ",\n  \"ppm_write_s\": " << metrics_.ppm_write_s
      << ",\n  \"ppm_fsync_s\": " << metrics_.ppm_fsync_s
      << ",\n  \"ppm_bytes\": " << metrics_.ppm_bytes
      << ",\n  \"ppm_metrics_bytes\": " << metrics_.ppm_metrics_bytes
      << ",\n  \"ppm_host_bytes\": " << metrics_.ppm_host_bytes
      << ",\n  \"ppm_frames\": " << metrics_.ppm_frames
      << ",\n  \"ppm_images\": " << metrics_.ppm_images
      << ",\n  \"maps_update_s\": " << (maps_?maps_->metrics().update_s:0)
      << ",\n  \"maps_copy_s\": " << (maps_?maps_->metrics().copy_s:0)
      << ",\n  \"maps_write_s\": " << (maps_?maps_->metrics().write_s:0)
      << ",\n  \"maps_render_s\": " << (maps_?maps_->metrics().render_s:0)
      << ",\n  \"maps_memory_bytes\": " << (maps_?maps_->metrics().memory_bytes:0)
      << ",\n  \"maps_reference_memory_bytes\": " << (maps_?maps_->metrics().reference_memory_bytes:0)
      << ",\n  \"maps_bytes\": " << (maps_?maps_->metrics().bytes:0)
      << ",\n  \"total_output_bytes\": " << metrics_.snapshot_bytes + metrics_.sismo_bytes + metrics_.stats_bytes + metrics_.sismo_stats_bytes + metrics_.slice_bytes + metrics_.fft.output_bytes + metrics_.ppm_bytes + metrics_.ppm_metrics_bytes + (maps_?maps_->metrics().bytes:0)
      << ",\n  \"raw_output_bytes\": " << metrics_.snapshot_bytes + metrics_.sismo_bytes
      << "\n}\n";
  out.close();
  if (!out) throw std::runtime_error("Cannot write metrics: " + metrics_file_);
}

// sismos : lit le fichier des recepteurs.
// Une ligne = un recepteur "x,y,z" (virgules, espaces ou ';' acceptes).
// Les lignes vides, les commentaires '#' et un eventuel en-tete texte
// sont ignores.
std::vector<std::array<float, 3>> SEMproxy::readReceivers(
    const std::string& filename)
{
  std::ifstream in(filename);
  if (!in) throw std::runtime_error("Cannot open receivers file: " + filename);
  std::vector<std::array<float, 3>> rcvs;
  std::string line;
  int lineNumber = 0;
  while (std::getline(in, line))
  {
    ++lineNumber;
    // 1. on enleve le commentaire eventuel (tout ce qui suit '#')
    auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    // 2. on remplace les separateurs par des espaces
    //    ('\r' : fin de ligne d'un fichier cree sous Windows)
    for (char& ch : line)
      if (ch == ',' || ch == ';' || ch == '\r') ch = ' ';
    // 3. on essaie de lire 3 nombres ; si la ligne est vide ou si c'est
    //    un en-tete texte, la lecture echoue et on passe a la suivante
    std::istringstream iss(line);
    std::array<float, 3> c;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    if (iss >> c[0] >> c[1] >> c[2]) {
      std::string extra;
      if (iss >> extra)
        throw std::runtime_error("Extra receiver column at line " + std::to_string(lineNumber));
      rcvs.push_back(c);
    } else {
      std::istringstream header(line);
      std::string x, y, z, extra;
      if (rcvs.empty() && (header >> x >> y >> z) && x == "x" && y == "y" && z == "z" && !(header >> extra))
        continue;
      throw std::runtime_error("Invalid receiver at " + filename + ":" + std::to_string(lineNumber));
    }
  }
  if (rcvs.empty())
    throw std::runtime_error("No receiver found in file: " + filename);
  return rcvs;
}
void SEMproxy::updateReceiverStatistics(int timestep)
{
  auto values = pnAtReceiver;
  auto stats = receiver_stats_device_;
  const int index = keep_receiver_traces_ ? timestep : 0;
  Kokkos::parallel_for("ReceiverWelford", rcv_coords_.size(), KOKKOS_LAMBDA(int r) {
    stats(r).add(values(r, index));
  });
}

void SEMproxy::saveReceiverStatistics()
{
  const auto copyStart = tp1::Clock::now();
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, receiver_stats_device_);
  Kokkos::fence();
  metrics_.sismo_stats_copy_s += tp1::seconds(copyStart);
  if (sismo_stats_file_.empty()) return;
  const auto writeStart = tp1::Clock::now();
  fs::path path(sismo_stats_file_);
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  std::ofstream out(path);
  out << "receiver,count,min,max,mean,variance\n" << std::setprecision(17);
  for (std::size_t r = 0; r < rcv_coords_.size(); ++r) {
    const auto& stats = host(r);
    out << r << ',' << stats.n << ',' << stats.minVal << ',' << stats.maxVal << ','
        << stats.mean << ',' << stats.variance() << '\n';
  }
  metrics_.sismo_stats_bytes = tp1::finish(out, path, durable_outputs_, metrics_.fsync_s);
  metrics_.sismo_stats_write_s += tp1::seconds(writeStart);
}

void SEMproxy::initSlice()
{
  const auto start = tp1::Clock::now();
  const auto plane = SliceUtils::selectPlane(*m_mesh, slice_plane_, slice_requested_);
  slice_coordinate_ = plane.coordinate;
  const auto& indices = plane.nodes;
  slice_nodes_ = allocateVector<vectorInt>(indices.size(), "sliceNodes");
  for (std::size_t n = 0; n < indices.size(); ++n) slice_nodes_(n) = indices[n];
  slice_values_ = decltype(slice_values_)("sliceValues", indices.size());
  slice_host_values_ = decltype(slice_host_values_)("sliceHostValues", indices.size());
  Kokkos::fence();
  metrics_.slice_setup_s = tp1::seconds(start);
}

void SEMproxy::saveSlice(int timestep)
{
  Kokkos::fence();
  const auto extractStart = tp1::Clock::now();
  SliceUtils::gather<Kokkos::DefaultExecutionSpace>(pnGlobal, i1, slice_nodes_, slice_values_);
  Kokkos::fence();
  metrics_.slice_extract_s += tp1::seconds(extractStart);
  const auto copyStart = tp1::Clock::now();
  Kokkos::deep_copy(slice_host_values_, slice_values_);
  Kokkos::fence();
  metrics_.slice_copy_s += tp1::seconds(copyStart);
  const auto writeStart = tp1::Clock::now();
  fs::create_directories(slice_folder_);
  auto finish = [&](std::ofstream& out, const fs::path& path) {
    metrics_.slice_bytes += tp1::finish(out, path, durable_outputs_, metrics_.fsync_s);
  };
  const int count = slice_nodes_.extent(0);
  if (!slice_coords_written_) {
    std::vector<float> xyz(3 * count);
    for (int n = 0; n < count; ++n)
      for (int d = 0; d < 3; ++d) xyz[3 * n + d] = m_mesh->nodeCoord(slice_nodes_(n), d);
    const auto path = fs::path(slice_folder_) / "coords.bin";
    std::ofstream out(path, std::ios::binary);
    tp1::writeFloat32(out, xyz);
    finish(out, path);
    slice_coords_written_ = true;
  }
  std::ostringstream name;
  name << "slice_it_" << std::setw(6) << std::setfill('0') << timestep;
  const auto base = fs::path(slice_folder_) / name.str();
  std::vector<float> data(count);
  for (int n = 0; n < count; ++n) data[n] = slice_host_values_(n);
  {
    const auto path = base.string() + ".bin";
    std::ofstream out(path, std::ios::binary);
    tp1::writeFloat32(out, data);
    finish(out, path);
  }
  const auto path = base.string() + ".json";
  std::ofstream meta(path);
  meta << std::setprecision(17)
       << "{\"schema\":\"funtides.snapshot.v1\",\"kind\":\"slice\",\"dtype\":\"<f4\",\"count\":" << count
       << ",\"coordinates\":\"coords.bin\",\"step\":" << timestep
       << ",\"dt_s\":" << dt_ << ",\"time_s\":" << (timestep + 1) * double(dt_)
       << ",\"plane\":" << tp1::jsonString(slice_plane_) << ",\"requested_position\":" << slice_requested_
       << ",\"position\":" << slice_coordinate_ << "}\n";
  finish(meta, path);
  ++metrics_.slice_count;
  metrics_.slice_write_s += tp1::seconds(writeStart);
}

void SEMproxy::computePressureStatistics(int column, int step)
{
  Kokkos::fence();
  const auto start = tp1::Clock::now();
  Kokkos::parallel_reduce("PressureWelford", m_mesh->getNumberOfNodes(),
                         PressureReduction{pnGlobal, column}, pressure_stats_device_);
  Kokkos::fence();
  const auto copied = tp1::Clock::now();
  const double computeMs = tp1::seconds(start) * 1000;
  Kokkos::deep_copy(pressure_stats_host_, pressure_stats_device_);
  Kokkos::fence();
  const double copyMs = tp1::seconds(copied) * 1000;
  const auto stats = pressure_stats_host_();
  recordStats(step, "gpu", stats.minVal, stats.maxVal, stats.mean, stats.variance(),
              copyMs, computeMs, copyMs + computeMs, sizeof(StreamStats));
}

void SEMproxy::computePressureStatisticsCPU(int column, int step)
{
  Kokkos::fence();
  const auto start = tp1::Clock::now();
  StreamStats stats;
  for (int n = 0; n < m_mesh->getNumberOfNodes(); ++n) stats.add(pnGlobal(n, column));
  const double computeMs = tp1::seconds(start) * 1000;
  recordStats(step, "cpu", stats.minVal, stats.maxVal, stats.mean, stats.variance(),
              0, computeMs, computeMs, 0);
}

void SEMproxy::computePressureStatisticsHostCopy(int column, int step)
{
  Kokkos::fence();
  const auto start = tp1::Clock::now();
  Kokkos::deep_copy(stats_host_column_, Kokkos::subview(pnGlobal, Kokkos::ALL, column));
  Kokkos::fence();
  const double copyMs = tp1::seconds(start) * 1000;
  const auto computeStart = tp1::Clock::now();
  StreamStats stats;
  for (int n = 0; n < m_mesh->getNumberOfNodes(); ++n) stats.add(stats_host_column_(n));
  const double computeMs = tp1::seconds(computeStart) * 1000;
  recordStats(step, "hostcopy", stats.minVal, stats.maxVal, stats.mean, stats.variance(),
              copyMs, computeMs, copyMs + computeMs,
              m_mesh->getNumberOfNodes() * sizeof(float));
}

void SEMproxy::recordStats(int step, const char* mode, double minP,
                           double maxP, double mean, double variance,
                           double copyMs, double computeMs, double totalMs,
                           std::size_t bytes)
{
  stats_total_ms_ += totalMs;
  stats_count_++;

  metrics_.stats_compute_s += computeMs / 1000;
  metrics_.stats_copy_s += copyMs / 1000;
  metrics_.stats_transfer_bytes += bytes;
  const auto writeStart = tp1::Clock::now();
  if (stats_out_.is_open())
    stats_out_ << step << ',' << minP << ',' << maxP << ',' << mean << ','
               << variance << ',' << totalMs << ',' << mode << ',' << copyMs
               << ',' << computeMs << ',' << bytes << ',' << (step + 1) * double(dt_)
               << ',' << m_mesh->getNumberOfNodes() << '\n';
  metrics_.stats_write_s += tp1::seconds(writeStart);
}

void SEMproxy::savePPMSlice(int step, int column)
{
    if (!visu_slice_) return;
    Kokkos::fence();
    const auto frameStart=tp1::Clock::now();

    std::ostringstream filename;
    filename << visu_folder_ << "/slice_" << visu_plane_ << "_it_"
             << std::setw(6) << std::setfill('0') << step;

    const TP3FrameMetrics metrics = visu_slice_->savePPM(
        pnGlobal, column, filename.str(), visu_format_, visu_scale_, visu_fsync_);

    metrics_.ppm_extract_s+=metrics.extract_ms/1000;
    metrics_.ppm_color_s+=metrics.color_ms/1000;
    metrics_.ppm_copy_s+=metrics.copy_ms/1000;
    metrics_.ppm_host_bytes+=metrics.host_bytes;
    ++metrics_.ppm_frames;
    for(const auto& file:metrics.files) {
      metrics_.ppm_bytes+=file.size_bytes; ++metrics_.ppm_images;
      metrics_.ppm_encode_s+=file.encode_ms/1000;
      metrics_.ppm_write_s+=file.write_ms/1000;
      metrics_.ppm_fsync_s+=file.fsync_ms/1000;
      metrics_.fsync_s+=file.fsync_ms/1000;
    }
    const std::filesystem::path csv_path =
        std::filesystem::path(visu_folder_) / "visu_metrics.csv";
    const bool write_header = !std::filesystem::exists(csv_path) ||
                              std::filesystem::file_size(csv_path) == 0;
    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) throw std::runtime_error("Cannot write visualization metrics CSV");
    csv << std::setprecision(12);
    if (write_header) {
        csv << "step,plane,format,width,height,size_bytes,extract_ms,color_ms,"
               "encode_ms,write_ms,fsync_ms,output_total_ms,frame_total_ms,"
               "fsync_enabled,path,backend,normalization,scale,max_abs,copy_ms,host_bytes,time_s,position_m\n";
    }
    for (const auto& file : metrics.files) {
        csv << step << ',' << visu_plane_ << ',' << file.format << ','
            << visu_width_ << ',' << visu_height_ << ',' << file.size_bytes << ','
            << metrics.extract_ms << ',' << metrics.color_ms << ','
            << file.encode_ms << ',' << file.write_ms << ',' << file.fsync_ms << ','
            << file.total_ms << ',' << metrics.total_ms << ','
            << (file.fsync_enabled ? 1 : 0) << ',' << file.path << ',' << visu_backend_ << ','
            << visu_normalization_ << ',' << metrics.scale << ',' << metrics.max_abs << ','
            << metrics.copy_ms << ',' << metrics.host_bytes << ',' << (step+1)*double(dt_) << ','
            << visu_slice_->planeCoordinate() << '\n';
    }
    if (!csv) throw std::runtime_error("Error writing visualization metrics CSV");
    metrics_.ppm_metrics_bytes=tp1::finish(csv,csv_path,visu_fsync_,metrics_.fsync_s);
    metrics_.ppm_s+=tp1::seconds(frameStart);
}

bool SEMproxy::runInSituProcessing(int step)
{
  if (maps_) maps_->update(pnGlobal,i1,step);

  // Steering is read first, so that its commands apply from this step on
  if (insitu_extra_ == "steering")
  {
    if (step % steer_interval_ == 0) steerFromFile(step);
    if (steer_stop_) return false;  // clean stop of the time loop
  }

  if (insitu_viz_ == "ppm")
  {
    if (step % visu_interval_ == 0)
    {
      savePPMSlice(step, i1);
    }
  }


  // Part 2: compression
  if (insitu_compress_ != "none")
  {
    // Lossless stages (RLE / DELTA): CPU pipeline of compressSnapshot()
    if (compress_cfg_.rle || compress_cfg_.delta)
    {
      if (step % snap_time_interval_ == 0) compressSnapshot(step);
    }
    // Quantization only (optionally on the GPU): global bounds,
    // validation, .cpm files
    else if (compress_cfg_.enabled && is_snapshots_ &&
             step % snap_time_interval_ == 0)
    {
      using Clock = std::chrono::steady_clock;
      const int nbNodes = m_mesh->getNumberOfNodes();
      const std::size_t rawBytes =
          static_cast<std::size_t>(nbNodes) * sizeof(float);
      // 0 -> bornes par snapshot ; > 0 -> bornes globales
      const float amaxFixed =
          compress_cfg_.global_bounds ? compress_cfg_.amax : 0.0f;

      std::vector<uint8_t> buf;
      double quantMs = 0.0;       // calcul de la quantification
      double copyMs = 0.0;        // transfert GPU -> hote (de ce que l'hote lit pour quantifier)
      uint64_t clipped = 0;       // valeurs saturees (bornes globales trop petites)
      float amaxUsed = 0.0f;
      std::size_t hostBytes = 0;  // octets lus par l'hote depuis le GPU

      Kokkos::fence();

      // ---- quantification sur le GPU : l'hote ne lit que 1 ou 2 octets/noeud
      if (compress_cfg_.gpu)
      {
        cmp_gpu::Timings tm;
        if (!cmp_gpu::quantize_gpu(pnGlobal, i2, nbNodes, compress_cfg_.nbits,
                                   step, amaxFixed, buf, clipped, amaxUsed, tm))
          throw std::runtime_error("quantize_gpu a echoue");
        quantMs = tm.reduce_ms + tm.quant_ms;
        copyMs = tm.copy_ms;
        hostBytes = static_cast<std::size_t>(nbNodes) * (compress_cfg_.nbits / 8);
      }

      // ---- copie de la colonne float sur l'hote : necessaire pour la version
      // CPU, pour la validation (compare) et pour le test octet par octet.
      // Elle n'est comptee dans copyMs/hostBytes que pour la version CPU.
      const bool needHostCol =
          !compress_cfg_.gpu || compress_cfg_.validate || compress_cfg_.check;
      Kokkos::View<float*, Layout, Kokkos::HostSpace> col;
      if (needHostCol)
      {
        col = Kokkos::View<float*, Layout, Kokkos::HostSpace>("cpmCol", nbNodes);
        Kokkos::fence();
        const auto c0 = Clock::now();
        Kokkos::deep_copy(col, Kokkos::subview(pnGlobal, Kokkos::ALL, i2));
        const auto c1 = Clock::now();
        if (!compress_cfg_.gpu)
        {
          copyMs = std::chrono::duration<double, std::milli>(c1 - c0).count();
          hostBytes = rawBytes;
        }
      }

      // ---- quantification sur le CPU
      if (!compress_cfg_.gpu)
      {
        const auto t0 = Clock::now();
        if (!cmp::quantize(col.data(), nbNodes, compress_cfg_.mode,
                           compress_cfg_.nbits, step, buf, amaxFixed, &clipped,
                           &amaxUsed))
          throw std::runtime_error("quantize a echoue");
        quantMs = std::chrono::duration<double, std::milli>(Clock::now() - t0)
                      .count();
      }

      // ---- verification : sortie GPU == sortie CPU, octet par octet
      int identical = -1;  // -1 = non verifie
      if (compress_cfg_.gpu && compress_cfg_.check)
      {
        std::vector<uint8_t> ref;
        if (!cmp::quantize(col.data(), nbNodes, cmp::Mode::IntN,
                           compress_cfg_.nbits, step, ref, amaxFixed))
          throw std::runtime_error("quantize CPU (verification) a echoue");
        identical = (ref == buf) ? 1 : 0;
        if (!identical)
        {
          std::size_t k = 0;
          while (k < ref.size() && k < buf.size() && ref[k] == buf[k]) ++k;
          std::cerr << "[compress] ATTENTION: GPU != CPU au pas " << step
                    << " (tailles " << buf.size() << " / " << ref.size()
                    << ", premier octet different : " << k << ")\n";
        }
      }

      if (clipped > 0)
        std::cerr << "[compress] ATTENTION: " << clipped
                  << " valeurs saturees au pas " << step << " (amax=" << amaxUsed
                  << " trop petit ?)\n";

      // ---- ecriture du .cpm
      std::filesystem::create_directories(snap_folder_ + "/compressed");
      char name[256];
      snprintf(name, sizeof(name), "%s/compressed/snapshot_it_%06d.cpm",
               snap_folder_.c_str(), step);
      std::ofstream f(name, std::ios::binary);
      f.write(reinterpret_cast<const char*>(buf.data()), buf.size());

      // ---- validation : on reconstruit et on mesure l'erreur
      // (desactivable avec +novalid pour mesurer le cout de la compression seule)
      double rmse = std::nan(""), maxAbs = std::nan("");
      if (compress_cfg_.validate)
      {
        std::vector<float> rec;
        if (!cmp::dequantize(buf.data(), buf.size(), rec))
          throw std::runtime_error("dequantize a echoue");
        const cmp::ErrorStats e = cmp::compare(col.data(), rec.data(), nbNodes);
        rmse = e.rmse_norm;
        maxAbs = e.max_abs_norm;
      }

      if (!compress_out_.is_open())
      {
        compress_out_.open(snap_folder_ + "/compress_stats.csv");
        compress_out_ << std::setprecision(9);
        compress_out_ << "step,mode,n_nodes,bytes_raw,bytes_cpm,ratio,"
                         "rmse_norm,max_abs_norm,quantize_ms,"
                         "copy_ms,bounds,amax,n_clipped,host_bytes,impl,identical\n";
      }
      compress_out_ << step << ',' << insitu_compress_ << ',' << nbNodes << ','
                    << rawBytes << ',' << buf.size() << ','
                    << cmp::ratio(rawBytes, buf.size()) << ',' << rmse << ','
                    << maxAbs << ',' << quantMs << ',' << copyMs << ','
                    << (compress_cfg_.global_bounds ? "global" : "snapshot")
                    << ',' << amaxUsed << ',' << clipped << ',' << hostBytes
                    << ',' << (compress_cfg_.gpu ? "gpu" : "cpu") << ','
                    << identical << '\n';
    }
  }

  // Part 3: other processing (steering, driven outputs, accumulated maps)
  if (insitu_extra_ != "none")
  {
  }

  return true;  // returning false would request a clean stop
}
// Compress every seismogram (one trace per receiver) with the pipeline given
// by --insitu-compress. Each trace is quantized with its own scale (its
// maximum), unlike the snapshots where the scale follows the whole field, so
// weak signals are not flattened by the source peak.
// stored_bytes includes the 40-byte header; raw_bytes = samples * 4.
void SEMproxy::compressSeismograms()
{
  const int nRcv = static_cast<int>(rcv_coords_.size());
  const std::uint64_t ns = static_cast<std::uint64_t>(num_sample_);
  for (int r = 0; r < nRcv; r++)
  {
    std::vector<float> trace(ns);
    for (std::uint64_t t = 0; t < ns; t++)
      trace[t] = pnAtReceiver(r, static_cast<int>(t));

    std::vector<std::uint8_t> buf;
    if (!cmp::quantize(trace.data(), ns, compress_cfg_.mode, compress_cfg_.nbits,
                       0, buf))
      throw std::runtime_error("compressSeismograms: quantize failed");
    if (compress_cfg_.delta && !cmp::delta_apply(buf))
      throw std::runtime_error("compressSeismograms: delta_apply failed");
    if (compress_cfg_.rle && !cmp::rle_apply(buf))
      throw std::runtime_error("compressSeismograms: rle_apply failed");
    const std::uint64_t stored = buf.size();

    // Reconstruction error, same order as the snapshots (undo in reverse)
    std::vector<std::uint8_t> chk = buf;
    if (compress_cfg_.rle && !cmp::rle_undo(chk))
      throw std::runtime_error("compressSeismograms: rle_undo failed");
    if (compress_cfg_.delta && !cmp::delta_undo(chk))
      throw std::runtime_error("compressSeismograms: delta_undo failed");
    std::vector<float> rec;
    if (!cmp::dequantize(chk.data(), chk.size(), rec))
      throw std::runtime_error("compressSeismograms: dequantize failed");
    const cmp::ErrorStats err = cmp::compare(trace.data(), rec.data(), ns);

    // Leading exact zeros: the wave has not reached the receiver yet
    std::uint64_t zeros = 0;
    while (zeros < ns && trace[zeros] == 0.0f) zeros++;

    const double raw = static_cast<double>(ns) * sizeof(float);
    std::cout << "[compress-sismo] rcv=" << r << " raw_bytes=" << raw
              << " stored_bytes=" << stored << " ratio="
              << raw / static_cast<double>(stored)
              << " rmse_norm=" << err.rmse_norm
              << " max_abs_norm=" << err.max_abs_norm
              << " leading_zero_samples=" << zeros << "/" << ns << std::endl;
  }
}


// Steering: read the control file and apply its commands. Format: one
// "key=value" per line, '#' starts a comment. Supported keys:
//   snap_interval=N    change the snapshot frequency (N >= 1)
//   slice=AXIS:VALUE   restrict the CSV snapshots to the node plane closest to
//                      AXIS=VALUE (AXIS is x, y or z); "slice=off" restores
//                      full snapshots
//   snapshot_now=N     write a snapshot now, each time the value N changes
//   stop=1             request a clean stop of the time loop
// The time spent reading the file is measured (steer_total_ms_); an immediate
// snapshot is written after the timer has stopped, so it is not counted.
void SEMproxy::steerFromFile(int step)
{
  const auto t0 = std::chrono::steady_clock::now();
  bool snapNow = false;

  auto trim = [](const std::string& s)
  {
    const char* ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return std::string();
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
  };

  std::ifstream in(steer_file_);
  if (in)
  {
    std::string line;
    while (std::getline(in, line))
    {
      const auto hash = line.find('#');
      if (hash != std::string::npos) line.erase(hash);
      const auto eq = line.find('=');
      if (eq == std::string::npos) continue;
      const std::string key = trim(line.substr(0, eq));
      const std::string val = trim(line.substr(eq + 1));
      try
      {
        if (key == "snap_interval")
        {
          const int v = std::stoi(val);
          if (v >= 1 && v != snap_time_interval_)
          {
            std::cout << "[steering] step " << step << ": snapshot interval "
                      << snap_time_interval_ << " -> " << v << std::endl;
            snap_time_interval_ = v;
          }
        }
        else if (key == "slice")
        {
          int axis = -1;
          double value = 0.0;
          if (val != "off")
          {
            if (val.size() < 3 || val[1] != ':')
              throw std::invalid_argument("slice");
            const char a = val[0];
            axis = (a == 'x' || a == 'X') ? 0
                 : (a == 'y' || a == 'Y') ? 1
                 : (a == 'z' || a == 'Z') ? 2 : -1;
            if (axis < 0) throw std::invalid_argument("slice axis");
            value = std::stod(val.substr(2));
          }
          if (axis != slice_axis_ || (axis >= 0 && value != slice_value_))
          {
            std::cout << "[steering] step " << step << ": slice -> "
                      << (axis < 0 ? std::string("off") : val) << std::endl;
            slice_axis_ = axis;
            slice_value_ = value;
          }
        }
        else if (key == "snapshot_now")
        {
          const int v = std::stoi(val);
          if (v != steer_last_now_)
          {
            steer_last_now_ = v;
            snapNow = true;
          }
        }
        else if (key == "stop")
        {
          if (std::stoi(val) == 1 && !steer_stop_)
          {
            std::cout << "[steering] step " << step << ": clean stop requested"
                      << std::endl;
            steer_stop_ = true;
          }
        }
      }
      catch (const std::exception&)
      {
        std::cerr << "[steering] ignored invalid line: " << key << "=" << val
                  << std::endl;
      }
    }
  }

  steer_total_ms_ += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
  steer_reads_++;

  if (snapNow)
  {
    std::cout << "[steering] step " << step << ": immediate snapshot"
              << std::endl;
    if (!compress_no_csv_) saveSnapshot(step);
    // the regular compression at this step (if any) is done by the hook
    if (insitu_compress_ != "none" && step % snap_time_interval_ != 0)
      compressSnapshot(step);
  }
}
// Compress the newest pressure field (column i1) with the pipeline given by
// --insitu-compress: quantize (raw, f16, int8, int16), then RLE if requested.
// Logs the raw size, the stored size, the ratio and the time in the CSV.
void SEMproxy::compressSnapshot(int step)
{

  const std::uint64_t n =
      static_cast<std::uint64_t>(m_mesh->getNumberOfNodes());
  
  
  // Node traversal order. "scan" sorts the nodes by z, then y, then x, so that
  // consecutive values are spatial neighbours. The permutation is built once
  // (outside the timed section) and is deterministic: the decoder can rebuild
  // it from the mesh, it is not stored in the .cmp file.
  if ((compress_order_ == "scan" || compress_order_ == "scanz") && node_order_.empty())
  {
    node_order_.resize(n);
    std::iota(node_order_.begin(), node_order_.end(), 0);
    std::stable_sort(node_order_.begin(), node_order_.end(),
                     [this](int a, int b)
                     {
                       // scan: z is the slowest key (x varies fastest)
                       // scanz: x is the slowest key (z varies fastest)
                       for (int s = 0; s < 3; s++)
                       {
                         const int d = (compress_order_ == "scanz") ? s : 2 - s;
                         const auto ca = m_mesh->nodeCoord(a, d);
                         const auto cb = m_mesh->nodeCoord(b, d);
                         if (ca != cb) return ca < cb;
                       }
                       return false;
                     });

     std::uint64_t moved = 0;
    for (std::uint64_t k = 0; k < n; k++)
      if (node_order_[k] != static_cast<int>(k)) moved++;
    std::cout << "[compress] scan order built: " << moved << " of " << n
              << " nodes change position" << std::endl;
  }
  const auto t0 = std::chrono::steady_clock::now();

  // Copy the column into a contiguous host buffer (works with both layouts)
  Kokkos::fence();
  std::vector<float> field(n);
  for (std::uint64_t k = 0; k < n; k++)
    field[k] = pnGlobal(static_cast<int>(k), i1);



  // Field in traversal order (the original order for "natural")
  std::vector<float> ordered;
  if (!node_order_.empty())
  {
    ordered.resize(n);
    for (std::uint64_t k = 0; k < n; k++) ordered[k] = field[node_order_[k]];
  }
  const std::vector<float>& toCompress = node_order_.empty() ? field : ordered;
  // Header + (quantized) payload
  std::vector<std::uint8_t> buf;
  if (!cmp::quantize(toCompress.data(), n, compress_cfg_.mode, compress_cfg_.nbits,
                     static_cast<std::uint32_t>(step), buf))
    throw std::runtime_error("compressSnapshot: quantize failed");

  
    // Lossless stages on the payload: DELTA first, then RLE
  if (compress_cfg_.delta && !cmp::delta_apply(buf))
    throw std::runtime_error("compressSnapshot: delta_apply failed");
  if (compress_cfg_.rle && !cmp::rle_apply(buf))
    throw std::runtime_error("compressSnapshot: rle_apply failed");
  const double timeMs = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
   
  
  // Reconstruction error, computed after the timer (not part of time_ms):
  // undo the RLE on a copy, dequantize, and compare with the original field.
  std::vector<std::uint8_t> chk = buf;
  if (compress_cfg_.rle && !cmp::rle_undo(chk))
    throw std::runtime_error("compressSnapshot: rle_undo failed");
  if (compress_cfg_.delta && !cmp::delta_undo(chk))
    throw std::runtime_error("compressSnapshot: delta_undo failed");
  std::vector<float> rec;
  if (!cmp::dequantize(chk.data(), chk.size(), rec))
    throw std::runtime_error("compressSnapshot: dequantize failed");

   // Back to the natural node order, so that the error, the statistics and the
  // receiver value are computed on comparable arrays
  if (!node_order_.empty())
  {
    std::vector<float> natural(n);
    for (std::uint64_t k = 0; k < n; k++) natural[node_order_[k]] = rec[k];
    rec.swap(natural);
  }
  const cmp::ErrorStats err = cmp::compare(field.data(), rec.data(), n);
  
  // Statistics of a field (single pass, Welford), used to compare the original
  // field with the reconstructed one: do the TP2 conclusions still hold?
  struct FieldStats { double mn, mx, mean, var; };
  auto fieldStats = [n](const std::vector<float>& v) {
    FieldStats s{v[0], v[0], 0.0, 0.0};
    double m2 = 0.0;
    for (std::uint64_t k = 0; k < n; k++)
    {
      const double x = v[k];
      if (x < s.mn) s.mn = x;
      if (x > s.mx) s.mx = x;
      const double delta = x - s.mean;
      s.mean += delta / static_cast<double>(k + 1);
      m2 += delta * (x - s.mean);
    }
    s.var = m2 / static_cast<double>(n);
    return s;
  };
  const FieldStats so = fieldStats(field);
  const FieldStats sr = fieldStats(rec);
  const double amax = cmp::max_abs(field.data(), n);
  // differences normalized by max|p| (no point-wise relative error: the field
  // has many near-zero values); the variance is positive, so a relative error
  // makes sense there.
  const double dMin = amax > 0 ? (sr.mn - so.mn) / amax : 0.0;
  const double dMax = amax > 0 ? (sr.mx - so.mx) / amax : 0.0;
  const double dMean = amax > 0 ? (sr.mean - so.mean) / amax : 0.0;
  const double dVarRel = so.var > 0 ? (sr.var - so.var) / so.var : 0.0;
  
  // Same interpolation as the seismogram in run(), for receiver 0, applied to
  // the original and to the reconstructed field: does quantization destroy the
  // weak amplitudes seen by the receiver?
  double rcvOrig = 0.0;
  double rcvRec = 0.0;
  if (!rcv_coords_.empty())
  {
    const int order = m_mesh->getOrder();
    for (int i = 0; i < order + 1; i++)
    {
      for (int j = 0; j < order + 1; j++)
      {
        for (int k = 0; k < order + 1; k++)
        {
          const int nodeIdx = m_mesh->globalNodeIndex(rhsElementRcv[0], i, j, k);
          const int local = i + j * (order + 1) + k * (order + 1) * (order + 1);
          const double w = rhsWeightsRcv(0, local);
          rcvOrig += static_cast<double>(field[nodeIdx]) * w;
          rcvRec += static_cast<double>(rec[nodeIdx]) * w;
        }
      }
    }
  }
  const std::uint64_t rawBytes = n * sizeof(float);
  const std::uint64_t storedBytes = buf.size();

  // Write the compressed buffer next to the CSV snapshots and time the write
  std::filesystem::create_directories(snap_folder_);
  char binName[256];
  snprintf(binName, sizeof(binName), "%s/snapshot_it_%06d.cmp",
           snap_folder_.c_str(), step);
  const auto tw0 = std::chrono::steady_clock::now();
  {
    std::ofstream f(binName, std::ios::binary);
    if (!f)
      throw std::runtime_error("compressSnapshot: cannot open " +
                               std::string(binName));
    f.write(reinterpret_cast<const char*>(buf.data()),
            static_cast<std::streamsize>(buf.size()));
    if (!f)
      throw std::runtime_error("compressSnapshot: write error on " +
                               std::string(binName));
  }
  const double writeMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - tw0)
                             .count();

  // Size of the CSV snapshot written at the same step (0 if it is absent)
  char csvName[256];
  snprintf(csvName, sizeof(csvName), "%s/snapshot_it_%06d.csv",
           snap_folder_.c_str(), step);
  std::error_code ec;
  const std::uintmax_t csvSize = std::filesystem::file_size(csvName, ec);
  const std::uint64_t csvBytes =
      ec ? 0 : static_cast<std::uint64_t>(csvSize);

  if (compress_out_.is_open())
    compress_out_ << step << ',' << rawBytes << ',' << storedBytes << ','
                  << cmp::ratio(rawBytes, storedBytes) << ',' << timeMs << ','
                  << writeMs << ',' << csvBytes << ',' << err.rmse_norm << ','
                  << err.max_abs_norm << ',' << dMin << ',' << dMax << ','
                  << dMean << ',' << dVarRel << ',' << rcvOrig << ','
                  << rcvRec << '\n';
}
