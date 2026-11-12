// compression_gpu.hpp
// Quantification int8 / int16 directement sur le GPU (Kokkos).
//
// Header-only et TEMPLATE exprès : les KOKKOS_LAMBDA doivent etre dans une
// fonction libre (nvcc interdit les lambdas etendus dans une methode privee).
// A inclure depuis sem_proxy.cc, qui contient deja des KOKKOS_LAMBDA.
//
// Principe : l'hote ne lit que 1 ou 2 octets par noeud (plus un float si les
// bornes sont par snapshot) au lieu de 4 octets par noeud.
//
// Egalite octet par octet avec cmp::quantize (CPU) :
//   - amax : un max est exact quel que soit l'ordre de la reduction ;
//   - scale : calcule SUR L'HOTE avec la meme expression float que le CPU ;
//   - r = x / scale en double (IEEE, identique CPU/GPU) ;
//   - arrondi : round() = demi vers l'exterieur de zero, comme std::lround ;
//   - ecretage a +-qmax, puis stockage int8_t / int16_t ;
//   - l'en-tete est construit par cmp::pack_intn (meme code que le CPU).
// Necessite Kokkos >= 3.6 (Kokkos::round, Kokkos::isfinite, Kokkos::fabs).
#pragma once
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdint>
#include <vector>

#include "compression.hpp"

namespace cmp_gpu {

struct Timings {
  double reduce_ms = 0.0;  // max|p| sur le GPU (0 si bornes globales)
  double quant_ms  = 0.0;  // kernel de quantification
  double copy_ms   = 0.0;  // transfert GPU -> hote des entiers
};

// field    : tableau 2D (ex. pnGlobal), colonne `column` a quantifier
// n        : nombre de noeuds
// amax_fixed : 0 -> bornes par snapshot (reduction GPU) ; > 0 -> bornes globales
// out      : en-tete + payload, identique a cmp::quantize
// n_clipped, amax_used : voir compression.hpp
template <class View2D>
bool quantize_gpu(const View2D& field, int column, int n, uint8_t nbits,
                  uint32_t step, float amax_fixed, std::vector<uint8_t>& out,
                  uint64_t& n_clipped, float& amax_used, Timings& tm) {
  out.clear();
  n_clipped = 0;
  amax_used = 0.0f;
  tm = Timings{};
  if (nbits != 8 && nbits != 16) return false;
  if (n < 0) return false;
  if (!(amax_fixed >= 0.0f)) return false;  // refuse aussi NaN

  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  View2D f = field;  // copie de la vue (capture par valeur dans les kernels)
  const int col = column;

  // ---- 1. borne amax --------------------------------------------------
  Kokkos::fence();
  auto t0 = Clock::now();
  float amax = amax_fixed;
  if (!(amax > 0.0f)) {  // bornes par snapshot : max|p| sur le GPU
    float m = 0.0f;
    Kokkos::parallel_reduce(
        "cmp_maxabs", n,
        KOKKOS_LAMBDA(const int i, float& lmax) {
          const float a = Kokkos::fabs(f(i, col));
          if (a > lmax) lmax = a;  // un NaN est ignore, comme sur le CPU
        },
        Kokkos::Max<float>(m));
    Kokkos::fence();
    amax = (m > 0.0f) ? m : 0.0f;
  }
  auto t1 = Clock::now();
  tm.reduce_ms = (amax_fixed > 0.0f) ? 0.0 : ms(t0, t1);
  amax_used = amax;

  // scale : meme expression, meme precision que cmp::quantize (CPU)
  const long      qmax = (1L << (nbits - 1)) - 1;  // 127 ou 32767
  const float     scale =
      (amax > 0.0f) ? amax / static_cast<float>(qmax) : 1.0f;
  const long long qmax_ll = static_cast<long long>(qmax);
  const bool      is8 = (nbits == 8);

  // ---- 2. kernel de quantification ------------------------------------
  Kokkos::View<int8_t*>  q8("cmp_q8", is8 ? n : 0);
  Kokkos::View<int16_t*> q16("cmp_q16", is8 ? 0 : n);

  long long clipped = 0;
  auto t2 = Clock::now();
  Kokkos::parallel_reduce(
      "cmp_quantize_int", n,
      KOKKOS_LAMBDA(const int i, long long& c) {
        const double r =
            static_cast<double>(f(i, col)) / static_cast<double>(scale);
        long long q = 0;
        if (Kokkos::isfinite(r)) q = static_cast<long long>(Kokkos::round(r));
        if (q > qmax_ll) { q = qmax_ll; ++c; }
        else if (q < -qmax_ll) { q = -qmax_ll; ++c; }
        if (is8) q8(i) = static_cast<int8_t>(q);
        else     q16(i) = static_cast<int16_t>(q);
      },
      clipped);
  Kokkos::fence();
  auto t3 = Clock::now();
  tm.quant_ms = ms(t2, t3);
  n_clipped = static_cast<uint64_t>(clipped);

  // ---- 3. transfert vers l'hote : 1 ou 2 octets par noeud -------------
  std::vector<uint8_t> dummy;  // (pour garder le meme flux d'appel des deux cas)
  bool ok = false;
  auto t4 = Clock::now();
  if (is8) {
    Kokkos::View<int8_t*, Kokkos::HostSpace> h("cmp_h8", n);
    Kokkos::deep_copy(h, q8);
    Kokkos::fence();
    auto t5 = Clock::now();
    tm.copy_ms = ms(t4, t5);
    ok = cmp::pack_intn(h.data(), static_cast<uint64_t>(n), nbits, scale, step, out);
  } else {
    Kokkos::View<int16_t*, Kokkos::HostSpace> h("cmp_h16", n);
    Kokkos::deep_copy(h, q16);
    Kokkos::fence();
    auto t5 = Clock::now();
    tm.copy_ms = ms(t4, t5);
    ok = cmp::pack_intn(h.data(), static_cast<uint64_t>(n), nbits, scale, step, out);
  }
  return ok;
}

}  // namespace cmp_gpu