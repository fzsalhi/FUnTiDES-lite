#pragma once
#include <cxxopts.hpp>
#include <stdexcept>
#include <string>
#include "compression_config.hpp"

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
  // TP3 in-situ processing: one option per part ("none" = disabled)
  std::string insitu_viz = "none";       // part 1: in-situ visualization
  std::string insitu_compress = "none";  // part 2: compression
  std::string insitu_extra = "none";     // part 3: other processing
  std::string compress_file = "";  // CSV output of the compression ratios
  // Order in which the nodes are traversed before compression:
  // "natural" (mesh numbering) or "scan" (sorted by z, then y, then x)
  std::string compress_order = "natural";
    // 1: do not write the CSV snapshots when compression is enabled (the
  // compressed .cmp file replaces them); 0: write both
  int compress_no_csv = 0;
  // Steering: control file re-read every steer_interval steps
  std::string steer_file = "";
  int steer_interval = 10;
  std::string compress_bounds = "snapshot";  // snapshot | global
  float       compress_amax   = 0.f;         // borne fixe si bounds == global
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
      CompressConfig cc; std::string err;
    if (!parse_compress(insitu_compress, cc, err)) throw std::runtime_error(err);
    if (compress_bounds != "snapshot" && compress_bounds != "global")
      throw std::runtime_error("compress-bounds must be snapshot|global");
    if (compress_bounds == "global" && compress_amax <= 0.f)
      throw std::runtime_error("compress-amax must be > 0 with global bounds");
    if (cc.enabled && !is_snapshots)
      throw std::runtime_error("insitu-compress requires --is-snapshots");
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
        cxxopts::value<std::string>(o.fft_file))
        ("insitu-viz", "In-situ visualization mode (none = disabled)",
        cxxopts::value<std::string>(o.insitu_viz))(
        "insitu-compress", "In-situ compression mode (none = disabled)",
        cxxopts::value<std::string>(o.insitu_compress))(
        "insitu-extra", "Other in-situ processing (none = disabled)",
        cxxopts::value<std::string>(o.insitu_extra))
        ("compress-bounds", "Quantization bounds: snapshot|global",
          cxxopts::value<std::string>(o.compress_bounds))
        ("compress-amax", "Fixed max |p| used when compress-bounds=global",
          cxxopts::value<float>(o.compress_amax))
        ("compress-file", "Output CSV file for the in-situ compression ratios",
          cxxopts::value<std::string>(o.compress_file))
        ("compress-order", "Node traversal order before compression: natural|scan",
          cxxopts::value<std::string>(o.compress_order)->default_value("natural"))
        ("compress-no-csv", "1: replace the CSV snapshots by the compressed ones",
          cxxopts::value<int>(o.compress_no_csv)->default_value("0"))        
("steer-file", "Steering control file, re-read during the run",
          cxxopts::value<std::string>(o.steer_file))
        ("steer-interval", "Read the steering file every N time steps",
          cxxopts::value<int>(o.steer_interval)->default_value("10"));

 }  
};
