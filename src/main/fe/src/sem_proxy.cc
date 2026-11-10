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
#include <cmath>
#include <algorithm>
#ifdef SEMPROXY_ENABLE_CUFFT
void computeSismoFFTcuFFT(const std::vector<float>& signals,
                          int nReceivers,
                          int nSamples,
                          float dt,
                          const std::string& fftFile);
#endif

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
  sismo_file_ = opt.sismo_file;
  fft_file_ = opt.fft_file;
  stats_interval_ = opt.stats_interval;
    if (!opt.stats_file.empty())
  {
    stats_out_.open(opt.stats_file);
    if (!stats_out_)
      throw std::runtime_error("Cannot open stats file: " + opt.stats_file);
    stats_out_ << std::setprecision(9);
    stats_out_ << "step,min,max,mean,variance,time_ms,mode,copy_ms,compute_ms,bytes\n";
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
  if (compress_cfg_.delta || compress_cfg_.rle)
    throw std::runtime_error("delta/rle pas encore disponibles (personne 3)");
  insitu_extra_ = opt.insitu_extra;
  visu_plane_ = opt.visu_plane;
  visu_position_ = opt.visu_position;
  visu_width_ = opt.visu_width;
  visu_height_ = opt.visu_height;
  visu_format_ = opt.visu_format;
  visu_interval_ = opt.visu_interval;
  visu_folder_ = opt.visu_folder;
  visu_scale_ = opt.visu_scale;
  visu_fsync_ = opt.visu_fsync;

  if (insitu_viz_ != "none" && insitu_viz_ != "ppm")
      throw std::runtime_error("Unknown visualization mode: " + insitu_viz_);
  // one streaming accumulator per receiver (rcv_coords_ is already filled here)
  if (stats_interval_ > 0)
    sismo_stats_.assign(rcv_coords_.size(), StreamStats());

  std::cout << "Number of receivers is " << rcv_coords_.size() << std::endl;
  bool isModelOnNodes = opt.isModelOnNodes;
  isElastic_ = opt.isElastic;
  is_snapshots_ = opt.is_snapshots;
  snap_time_interval_ = opt.snap_time_interval;
  snap_folder_ = opt.snap_folder;
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
  if (insitu_viz_ == "ppm")
  {
      std::filesystem::create_directories(visu_folder_);

      visu_slice_ = std::make_unique<TP3SliceCPU>(
          *m_mesh,
          nb_elements_,
          domain_size_,
          visu_plane_,
          visu_position_,
          visu_width_,
          visu_height_
      );
  }
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
  num_sample_ = timemax_ / dt_;
  m_solver = SolverFactory::createSolver(methodType, implemType, meshType,
                                         modelLocation, physicType, order);
  m_solver->computeFEInit(*m_mesh, sponge_size, opt.surface_sponge,
                          opt.taper_delta);
  initFiniteElem();
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
void SEMproxy::run()
{
  time_point<system_clock> startComputeTime, startOutputTime, startSaveTime,
      totalComputeTime, totalOutputTime, totalSaveTime;
  SEMsolverDataAcoustic solverData(i1, i2, myRHSTerm, pnGlobal, rhsElement,
                                   rhsWeights);
  for (int indexTimeSample = 0; indexTimeSample < num_sample_;
       indexTimeSample++)
  {
    startComputeTime = system_clock::now();
    m_solver->computeOneStep(dt_, indexTimeSample, solverData);
    totalComputeTime += system_clock::now() - startComputeTime;
    startOutputTime = system_clock::now();
    if (indexTimeSample % 50 == 0)
    {
      m_solver->outputSolutionValues(indexTimeSample, i1, rhsElement[0],
                                     pnGlobal, "pnGlobal");
    }
    // Save pressure at each receiver
    const int order = m_mesh->getOrder();
    // sismos : meme calcul qu'avant, repete pour chaque recepteur r
    for (int r = 0; r < (int)rcv_coords_.size(); r++)
    {
      float varnp1 = 0.0;
      for (int i = 0; i < order + 1; i++)
      {
        for (int j = 0; j < order + 1; j++)
        {
          for (int k = 0; k < order + 1; k++)
          {
            // noeud (i,j,k) de la maille du recepteur r
            int nodeIdx = m_mesh->globalNodeIndex(rhsElementRcv[r], i, j, k);
            int globalNodeOnElement =
                i + j * (order + 1) + k * (order + 1) * (order + 1);
            // pression du noeud x poids d'interpolation du recepteur r
            varnp1 += pnGlobal(nodeIdx, i2) *
                      rhsWeightsRcv(r, globalNodeOnElement);
          }
        }
      }
      // row r = seismogram of receiver r, column = time step
      pnAtReceiver(r, indexTimeSample) = varnp1;
      // update at every time step: the statistics must cover the whole signal
      if (stats_interval_ > 0) sismo_stats_[r].add(varnp1);
    }

          
      if (is_snapshots_ && indexTimeSample % snap_time_interval_ == 0)         
    {
      startSaveTime = system_clock::now();
      saveSnapshot(indexTimeSample);
      totalSaveTime += system_clock::now() - startSaveTime;
    }
    // statistiques in situ : apres computeOneStep, i1 contient le champ le plus recent
    if (stats_interval_ > 0 && indexTimeSample % stats_interval_ == 0)
      {
       if (stats_mode_ == "gpu")
        computePressureStatistics(i1, indexTimeSample);
      else if (stats_mode_ == "hostcopy")
        computePressureStatisticsHostCopy(i1, indexTimeSample);
      else
        computePressureStatisticsCPU(i1, indexTimeSample);
     }
     // TP3: single entry point of the in-situ processing. At this point,
    // column i1 holds the newest field. A false return means a clean stop.
    if (!runInSituProcessing(indexTimeSample)) break;
    swap(i1, i2);
    auto tmp = solverData.m_i1;
    solverData.m_i1 = solverData.m_i2;
    solverData.m_i2 = tmp;
    totalOutputTime += system_clock::now() - startOutputTime;
  }
  #ifdef SEMPROXY_ENABLE_CUFFT
  if (!fft_file_.empty())
  {
    const int nReceivers =
        static_cast<int>(rcv_coords_.size());

    std::vector<float> signals(
        static_cast<std::size_t>(nReceivers) *
        static_cast<std::size_t>(num_sample_));

    for (int r = 0; r < nReceivers; ++r)
    {
      for (int t = 0; t < num_sample_; ++t)
      {
        signals[
            static_cast<std::size_t>(r) * num_sample_ + t] =
            pnAtReceiver(r, t);
      }
    }

    computeSismoFFTcuFFT(
        signals,
        nReceivers,
        num_sample_,
        dt_,
        fft_file_);
  }
  #endif
  float kerneltime_ms = time_point_cast<microseconds>(totalComputeTime)
                            .time_since_epoch()
                            .count();
  float outputtime_ms =
      time_point_cast<microseconds>(totalOutputTime).time_since_epoch().count();
  cout << "------------------------------------------------ " << endl;
  cout << "\n---- Elapsed Kernel Time : " << kerneltime_ms / 1E6 << " seconds."
       << endl;
  cout << "---- Elapsed Output Time : " << outputtime_ms / 1E6 << " seconds."
       << endl;    
    // final in-situ statistics of each seismogram
  if (stats_interval_ > 0)
    for (size_t r = 0; r < sismo_stats_.size(); r++)
      cout << "[In-situ sismo] rcv=" << r << " n=" << sismo_stats_[r].n
           << " min=" << sismo_stats_[r].minVal
           << " max=" << sismo_stats_[r].maxVal
           << " moyenne=" << sismo_stats_[r].mean
           << " variance=" << sismo_stats_[r].variance() << endl;
    cout << "---- Elapsed Stats Time : " << stats_total_ms_ / 1E3
         << " seconds (" << stats_count_ << " calls)." << endl;
    
  cout << "------------------------------------------------ " << endl;
  // Save all receiver seismograms to CSV
  if (!sismo_file_.empty())
  {
    startSaveTime = system_clock::now();
    bool sismoSaved = false;
    {
      std::ofstream sismoFile(sismo_file_);
      if (!sismoFile)
      {
        std::cerr << "Error: cannot open sismo file: "
                  << sismo_file_ << std::endl;
      }
      else
      {
        sismoFile << "time";
        for (int r = 0; r < (int)rcv_coords_.size(); r++)
        {
          sismoFile << ",receiver_" << r;
        }
        sismoFile << '\n';
        sismoFile << std::setprecision(9);
        for (int t = 0; t < num_sample_; t++)
        {
          sismoFile << t * dt_;
          for (int r = 0; r < (int)rcv_coords_.size(); r++)
          {
            sismoFile << "," << pnAtReceiver(r, t);
          }
          sismoFile << '\n';
        }
        sismoSaved = true;
      }
    }
    totalSaveTime += system_clock::now() - startSaveTime;
    if (sismoSaved)
      std::cout << "Seismograms saved to " << sismo_file_ << std::endl;
  }
  float savetime_ms =
      time_point_cast<microseconds>(totalSaveTime).time_since_epoch().count();
  cout << "---- Elapsed Save Time : " << savetime_ms / 1E6 << " seconds."
       << endl;
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
  pnAtReceiver = allocateArray2D<arrayReal>(nRcv, num_sample_, "pnAtReceiver");
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
  int lx = domain_size_[0];
  int ly = domain_size_[1];
  int lz = domain_size_[2];
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
    int receiver_index = floor((rcv_coords_[r][0] * ex) / lx) +
                         floor((rcv_coords_[r][1] * ey) / ly) * ex +
                         floor((rcv_coords_[r][2] * ez) / lz) * ey * ex;
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
void SEMproxy::saveSnapshot(int timestep)
{
  const int nbNodes = m_mesh->getNumberOfNodes();
  std::filesystem::create_directories(snap_folder_);
  char filename[256];
  snprintf(filename, sizeof(filename), "%s/snapshot_it_%06d.csv",
           snap_folder_.c_str(), timestep);
  std::ofstream file(filename);
  if (!file.is_open())
    throw std::runtime_error("Impossible d'ouvrir le fichier snapshot CSV.");
  file << "x,y,z,pressure\n";
  file << std::setprecision(9);
  for (int n = 0; n < nbNodes; ++n)
  {
    file << m_mesh->nodeCoord(n, 0) << ','
         << m_mesh->nodeCoord(n, 1) << ','
         << m_mesh->nodeCoord(n, 2) << ','
         << pnGlobal(n, i2) << '\n';
  }
  if (!file)
    throw std::runtime_error("Erreur pendant l'ecriture du snapshot CSV.");
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
  while (std::getline(in, line))
  {
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
    if (iss >> c[0] >> c[1] >> c[2]) rcvs.push_back(c);
  }
  if (rcvs.empty())
    throw std::runtime_error("No receiver found in file: " + filename);
  return rcvs;
}
void SEMproxy::computePressureStatistics(int column, int step)
{
  const int nbNodes = m_mesh->getNumberOfNodes();

  if (nbNodes <= 0)
    throw std::runtime_error("Cannot compute statistics on an empty field.");
  
  Kokkos::fence();

  const auto start = std::chrono::steady_clock::now();

  auto pressure = pnGlobal;

  float minPressure = std::numeric_limits<float>::max();
  float maxPressure = std::numeric_limits<float>::lowest();
  double sumPressure = 0.0;

  Kokkos::parallel_reduce(
      "PressureMin",
      nbNodes,
      KOKKOS_LAMBDA(const int n, float& localMin) {
        const float value = pressure(n, column);
        if (value < localMin)
          localMin = value;
      },
      Kokkos::Min<float>(minPressure));

  Kokkos::parallel_reduce(
      "PressureMax",
      nbNodes,
      KOKKOS_LAMBDA(const int n, float& localMax) {
        const float value = pressure(n, column);
        if (value > localMax)
          localMax = value;
      },
      Kokkos::Max<float>(maxPressure));

  Kokkos::parallel_reduce(
      "PressureSum",
      nbNodes,
      KOKKOS_LAMBDA(const int n, double& localSum) {
        localSum += static_cast<double>(pressure(n, column));
      },
      sumPressure);

  const double meanPressure = sumPressure / nbNodes;
  double variancePressure = 0.0;

  Kokkos::parallel_reduce(
      "PressureVariance",
      nbNodes,
      KOKKOS_LAMBDA(const int n, double& localVariance) {
        const double difference =
            static_cast<double>(pressure(n, column)) - meanPressure;
        localVariance += difference * difference;
      },
      variancePressure);

  variancePressure /= nbNodes;

  Kokkos::fence();

  const double elapsedMs =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();

  recordStats(step, "gpu", minPressure, maxPressure, meanPressure,
              variancePressure, 0.0, elapsedMs, elapsedMs,
              2 * sizeof(float) + 2 * sizeof(double));

  std::cout << "[In-situ Kokkos] pas=" << step
            << " min=" << minPressure
            << " max=" << maxPressure
            << " moyenne=" << meanPressure
            << " variance=" << variancePressure
            << " temps_ms=" << elapsedMs << '\n';
}
// Statistiques in situ du champ de pression, version CPU, une seule passe.
// column : colonne de pnGlobal a analyser (celle qui vient d'etre calculee)
// step   : numero du pas de temps (pour l'affichage uniquement)
void SEMproxy::computePressureStatisticsCPU(int column, int step)
{
  const int nbNodes = m_mesh->getNumberOfNodes();
  if (nbNodes <= 0)
    throw std::runtime_error("Cannot compute statistics on an empty field.");

  Kokkos::fence();  // make sure previous solver kernels are done before timing
  const auto start = std::chrono::steady_clock::now();

  double minP = pnGlobal(0, column);
  double maxP = minP;
  double mean = 0.0;  // moyenne courante
  double m2 = 0.0;    // cumul des carres des ecarts a la moyenne courante

  for (int n = 0; n < nbNodes; n++)
  {
    const double x = pnGlobal(n, column);
    if (x < minP) minP = x;
    if (x > maxP) maxP = x;
    // Welford : mise a jour incrementale
    const double delta = x - mean;
    mean += delta / (n + 1);
    m2 += delta * (x - mean);
  }
  const double variance = m2 / nbNodes;  // variance de population

  const double elapsedMs =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start)
          .count();
  recordStats(step, "cpu", minP, maxP, mean, variance, 0.0, elapsedMs,
              elapsedMs, 0);

  std::cout << "[In-situ CPU] pas=" << step << " min=" << minP
            << " max=" << maxP << " moyenne=" << mean
            << " variance=" << variance << " temps_ms=" << elapsedMs << '\n';
}

// In-situ statistics, host version: explicit GPU->host copy of one column,
// then a single-pass CPU loop (Welford). Copy and compute are timed separately.
// column : column of pnGlobal to analyse (the one just computed)
// step   : time step number (display only)
void SEMproxy::computePressureStatisticsHostCopy(int column, int step)
{
  const int nbNodes = m_mesh->getNumberOfNodes();
  if (nbNodes <= 0)
    throw std::runtime_error("Cannot compute statistics on an empty field.");

  // Host buffer, allocated outside the timed region
  Kokkos::View<float*, Layout, Kokkos::HostSpace> hostCol("hostCol", nbNodes);
  // The column is contiguous because Layout is LayoutLeft with CUDA
  auto col = Kokkos::subview(pnGlobal, Kokkos::ALL, column);

  Kokkos::fence();  // make sure the solver kernels are done before timing
  const auto t0 = std::chrono::steady_clock::now();
  Kokkos::deep_copy(hostCol, col);  // the GPU -> host transfer
  const auto t1 = std::chrono::steady_clock::now();

  // Single-pass statistics on the host copy
  double minP = hostCol(0);
  double maxP = minP;
  double mean = 0.0;  // running mean
  double m2 = 0.0;    // sum of squared deviations from the running mean

  for (int n = 0; n < nbNodes; n++)
  {
    const double x = hostCol(n);
    if (x < minP) minP = x;
    if (x > maxP) maxP = x;
    // Welford: incremental update
    const double delta = x - mean;
    mean += delta / (n + 1);
    m2 += delta * (x - mean);
  }
  const double variance = m2 / nbNodes;

  const auto t2 = std::chrono::steady_clock::now();

  const double copyMs =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double computeMs =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  const std::size_t bytes = static_cast<std::size_t>(nbNodes) * sizeof(float);
  
  recordStats(step, "hostcopy", minP, maxP, mean, variance, copyMs,
              computeMs, copyMs + computeMs, bytes);
  
  std::cout << "[In-situ HostCopy] step=" << step << " min=" << minP
            << " max=" << maxP << " mean=" << mean
            << " variance=" << variance << " copy_ms=" << copyMs
            << " compute_ms=" << computeMs << " bytes=" << bytes << '\n';
}

void SEMproxy::recordStats(int step, const char* mode, double minP,
                           double maxP, double mean, double variance,
                           double copyMs, double computeMs, double totalMs,
                           std::size_t bytes)
{
  stats_total_ms_ += totalMs;
  stats_count_++;

  if (stats_out_.is_open())
    stats_out_ << step << ',' << minP << ',' << maxP << ',' << mean << ','
               << variance << ',' << totalMs << ',' << mode << ',' << copyMs
               << ',' << computeMs << ',' << bytes << '\n';
}

void SEMproxy::savePPMSlice(int step, int column)
{
    if (!visu_slice_) return;
    Kokkos::fence();

    std::ostringstream filename;
    filename << visu_folder_ << "/slice_" << visu_plane_ << "_it_"
             << std::setw(6) << std::setfill('0') << step;

    const TP3FrameMetrics metrics = visu_slice_->savePPM(
        pnGlobal, column, filename.str(), visu_format_, visu_scale_, visu_fsync_);

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
               "fsync_enabled,path\n";
    }
    for (const auto& file : metrics.files) {
        csv << step << ',' << visu_plane_ << ',' << file.format << ','
            << visu_width_ << ',' << visu_height_ << ',' << file.size_bytes << ','
            << metrics.extract_ms << ',' << metrics.color_ms << ','
            << file.encode_ms << ',' << file.write_ms << ',' << file.fsync_ms << ','
            << file.total_ms << ',' << metrics.total_ms << ','
            << (file.fsync_enabled ? 1 : 0) << ',' << file.path << '\n';
    }
    if (!csv) throw std::runtime_error("Error writing visualization metrics CSV");
}

bool SEMproxy::runInSituProcessing(int step)
{
  
  // Part 1: in-situ visualization 
  if (insitu_viz_ == "ppm")
  {
    if (step % visu_interval_ == 0)
    {
      savePPMSlice(step, i2);
    }
  }


  // Part 2: compression
  if (insitu_compress_ != "none")
  {
    if (compress_cfg_.enabled && is_snapshots_ && step % snap_time_interval_ == 0)
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
