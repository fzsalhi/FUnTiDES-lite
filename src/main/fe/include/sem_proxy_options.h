#pragma once
#include <cxxopts.hpp>
#include <stdexcept>
#include <string>


class SemProxyOptions
{

 public:
  // Defaults
  int order = 2;
  int ex = 50, ey = 50, ez = 50;
  float lx = 2000.f, ly = 2000.f, lz = 2000.f;
  float srcx = 1010.f, srcy = 1010.f, srcz = 1010.f;
  float rcvx = 1410.f, rcvy = 1010.f, rcvz = 1010.f;
  //sismos: fichier d'entrée des récepteurs et CSV de sortie
  std::string rcv_file = "";
  std::string fft_file = "";
  std::string stats_file = "";
  // In-situ statistics mode: cpu | gpu | hostcopy
  std::string stats_mode = "cpu";
  int stats_interval = 0;
  std::string sismo_file = "sismos.csv";
  std::string implem = "makutu";  // makutu|shiva
  std::string method = "sem";     // sem|dg
  std::string mesh = "cartesian";
  float dt = 0.001;
  float timemax = 1.5;
  bool autodt = false;
  // sponge boundaries parameters
  float boundaries_size = 0;
  bool surface_sponge = false;
  float taper_delta = 0.015;
  // Boolean to tell if the model is charged on nodes or on element
  bool isModelOnNodes = false;
  bool isElastic = false;
  bool is_snapshots = false;
  int snap_time_interval = 50;
  std::string snap_folder = "../data";
  void validate() const
  {
    if (order < 1) throw std::runtime_error("order must be >= 1");
    if (ex <= 0 || ey <= 0 || ez <= 0)
      throw std::runtime_error("ex/ey/ez must be > 0");
    if (lx <= 0 || ly <= 0 || lz <= 0)
      throw std::runtime_error("lx/ly/lz must be > 0");
    if (is_snapshots && snap_time_interval <= 0)
      throw std::runtime_error("snap-time-interval must be > 0");
  }
  // Bind CLI flags to this instance (no --help here)
  static void bind_cli(cxxopts::Options& opts, SemProxyOptions& o)
  {
    opts.add_options()("o,order", "Order of approximation",
                       cxxopts::value<int>(o.order))(
        "ex", "Number of elements on X (Cartesian mesh)",
        cxxopts::value<int>(o.ex))("ey",
                                   "Number of elements on Y (Cartesian mesh)",
                                   cxxopts::value<int>(o.ey))(
        "ez", "Number of elements on Z (Cartesian mesh)",
        cxxopts::value<int>(o.ez))("lx", "Domain size X (Cartesian)",
                                   cxxopts::value<float>(o.lx))(
        "ly", "Domain size Y (Cartesian)", cxxopts::value<float>(o.ly))(
        "lz", "Domain size Z (Cartesian)", cxxopts::value<float>(o.lz))(
        "implem", "Implementation: makutu|shiva",
        cxxopts::value<std::string>(o.implem))(
        "method", "Method: sem|dg", cxxopts::value<std::string>(o.method))(
        "mesh", "Mesh: cartesian|ucartesian",
        cxxopts::value<std::string>(o.mesh))(
        "dt", "Time step selection in s (default = 0.001s)",
        cxxopts::value<float>(o.dt))(
        "timemax", "Duration of the simulation in s (default = 1.5s)",
        cxxopts::value<float>(o.timemax))(
        "auto-dt", "Select automatique dt via CFL equation.",
        cxxopts::value<bool>(o.autodt))(
        "boundaries-size", "Size of absorbing boundaries (meters)",
        cxxopts::value<float>(o.boundaries_size))(
        "sponge-surface", "Considere the surface's nodes as non sponge nodes",
        cxxopts::value<bool>(o.surface_sponge))(
        "taper-delta", "Taper delta for sponge boundaries value",
        cxxopts::value<float>(o.taper_delta))(
        "is-model-on-nodes",
        "Boolean to tell if the model is charged on nodes (true) or on element "
        "(false)",
        cxxopts::value<bool>(o.isModelOnNodes))(
        "is-elastic", "Elastic simulation", cxxopts::value<bool>(o.isElastic))(
        "rcv-file", "Input file with receiver positions (one x,y,z per line)",
        cxxopts::value<std::string>(o.rcv_file))(
        "sismo-file", "Output CSV file for the sismos (default sismos.csv)",
        cxxopts::value<std::string>(o.sismo_file))(
        "is-snapshots", "Save pressure snapshots (binary)",
        cxxopts::value<bool>(o.is_snapshots))(
        "snap-time-interval", "Number of time steps between two snapshots",
        cxxopts::value<int>(o.snap_time_interval))
        ("snap-folder", "Output folder for snapshots",
        cxxopts::value<std::string>(o.snap_folder))
        ("stats-interval",
        "Number of time steps between two in-situ statistics (0 = disabled)",
        cxxopts::value<int>(o.stats_interval))
        ("stats-file",
        "Output CSV file for the in-situ statistics",
        cxxopts::value<std::string>(o.stats_file))
        ("stats-mode",
        "In-situ stats mode: cpu|gpu|hostcopy (default cpu)",
        cxxopts::value<std::string>(o.stats_mode))
        ("fft-file",
        "Output CSV file for the in-situ cuFFT spectra",
        cxxopts::value<std::string>(o.fft_file));
  }
};
