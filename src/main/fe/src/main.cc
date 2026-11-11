// SEM driver: steady-clock measurements include completed Kokkos work.
#include "sem_proxy.h"
#include "sem_proxy_options.h"
#include <cstdlib>

int main(int argc, char* argv[])
{
  const auto start = tp1::Clock::now();
  try {
    // Respect affinity selected by a batch scheduler or the benchmark runner.
    setenv("OMP_PROC_BIND", "spread", 0);
    setenv("OMP_PLACES", "cores", 0);
    Kokkos::ScopeGuard kokkos(argc, argv);
    cxxopts::Options options("SEM Proxy", "Runs the SEM simulation.");
    options.add_options()("h,help", "Print help message");
    SemProxyOptions opt;
    SemProxyOptions::bind_cli(options, opt);
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }
    opt.validate();
    SEMproxy simulation(opt);
    Kokkos::fence();
    const double initial = tp1::seconds(start);
    const auto runStart = tp1::Clock::now();
    simulation.run();
    Kokkos::fence();
    const double run = tp1::seconds(runStart);
    const double total = tp1::seconds(start);
    simulation.writeMetrics(initial, total);
    std::cout << "Elapsed Initial Time : " << initial << " seconds.\n"
              << "Elapsed Compute Time : " << run << " seconds.\n"
              << "Elapsed TotalExe Time : " << total << " seconds.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SEM error: " << error.what() << '\n';
    return 1;
  }
}
