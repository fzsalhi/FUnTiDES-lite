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
  stats_interval_ = opt.stats_interval;
    if (!opt.stats_file.empty())
  {
    stats_out_.open(opt.stats_file);
    if (!stats_out_)
      throw std::runtime_error("Cannot open stats file: " + opt.stats_file);
    stats_out_ << std::setprecision(9);
    stats_out_ << "step,min,max,mean,variance,time_ms\n";
  }

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
      // ligne r = sismo du recepteur r, colonne = pas de temps
      pnAtReceiver(r, indexTimeSample) = varnp1;
    }
    if (is_snapshots_ && indexTimeSample % snap_time_interval_ == 0)
    {
      startSaveTime = system_clock::now();
      saveSnapshot(indexTimeSample);
      totalSaveTime += system_clock::now() - startSaveTime;
    }
    // statistiques in situ : apres computeOneStep, i1 contient le champ le plus recent
    if (stats_interval_ > 0 && indexTimeSample % stats_interval_ == 0)
      computePressureStatisticsCPU(i1, indexTimeSample);
    swap(i1, i2);
    auto tmp = solverData.m_i1;
    solverData.m_i1 = solverData.m_i2;
    solverData.m_i2 = tmp;
    totalOutputTime += system_clock::now() - startOutputTime;
  }
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
  if (stats_interval_ > 0)
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
  stats_total_ms_ += elapsedMs;
  stats_count_++;

  if (stats_out_.is_open())
    stats_out_ << step << ',' << minP << ',' << maxP << ',' << mean << ','
               << variance << ',' << elapsedMs << '\n';

  std::cout << "[In-situ CPU] pas=" << step << " min=" << minP
            << " max=" << maxP << " moyenne=" << mean
            << " variance=" << variance << " temps_ms=" << elapsedMs << '\n';
}
