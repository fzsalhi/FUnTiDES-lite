// compression.cpp
#include "compression.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace cmp {
namespace {

// float32 -> binary16 (IEEE 754), arrondi au plus proche pair.
// Ecrit a la main pour ne pas dependre du support _Float16 du compilateur.
uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp  = (x >> 23) & 0xFFu;
    uint32_t       man  = x & 0x7FFFFFu;

    if (exp == 0xFFu) {  // infini ou NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    const int e = static_cast<int>(exp) - 127 + 15;  // exposant biaise half
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // debordement -> inf
    if (e <= 0) {                                               // sous-normal ou zero
        if (e < -10) return static_cast<uint16_t>(sign);
        man |= 0x800000u;  // bit implicite
        const int      shift    = 14 - e;
        uint32_t       half_man = man >> shift;
        const uint32_t rem      = man & ((1u << shift) - 1u);
        const uint32_t halfway  = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_man & 1u))) ++half_man;
        return static_cast<uint16_t>(sign | half_man);
    }
    uint32_t       half = sign | (static_cast<uint32_t>(e) << 10) | (man >> 13);
    const uint32_t rem  = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // la retenue passe dans l'exposant
    return static_cast<uint16_t>(half);
}

// binary16 -> float32 (exact).
float half_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t       man  = h & 0x3FFu;
    uint32_t       x;
    if (exp == 0) {
        if (man == 0) {
            x = sign;
        } else {  // sous-normal : on normalise
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            man &= 0x3FFu;
            x = sign | (static_cast<uint32_t>(112 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (man << 13);
    } else {
        x = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof f);
    return f;
}

bool valid_nbits(uint8_t nbits) { return nbits == 8 || nbits == 16; }

// Unique endroit ou l'en-tete est construit (CPU et GPU passent par ici).
Header make_header(Mode mode, uint8_t nbits, uint64_t n, uint64_t payload,
                   float scale, uint32_t step) {
    Header h{};  // initialise a zero : l'en-tete est identique octet par octet
    h.magic        = MAGIC;
    h.version      = VERSION;
    h.mode         = mode;
    h.nbits        = nbits;
    h.flags        = 0;
    h.n_elements   = n;
    h.payload_size = payload;
    h.scale        = scale;
    h.step         = step;
    return h;
}

}  // namespace

float max_abs(const float* x, uint64_t n) {
    float m = 0.0f;
    for (uint64_t i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > m) m = a;  // un NaN est ignore (comparaison fausse)
    }
    return m;
}

bool quantize(const float* in, uint64_t n, Mode mode, uint8_t nbits,
              uint32_t step, std::vector<uint8_t>& out,
              float amax_fixed, uint64_t* n_clipped, float* amax_used) {
    out.clear();
    if (n_clipped) *n_clipped = 0;
    if (amax_used) *amax_used = 0.0f;
    if (n > 0 && in == nullptr) return false;
    if (!std::isfinite(amax_fixed) || amax_fixed < 0.0f) return false;

    uint64_t elem = 0;  // octets par valeur
    switch (mode) {
        case Mode::Raw:  elem = 4; nbits = 0; break;
        case Mode::F16:  elem = 2; nbits = 0; break;
        case Mode::IntN:
            if (!valid_nbits(nbits)) return false;
            elem = nbits / 8;
            break;
        default: return false;
    }
    const uint64_t payload = n * elem;

    float    scale   = 1.0f;
    long     qmax    = 0;
    uint64_t clipped = 0;
    if (mode == Mode::IntN) {
        // bornes globales si amax_fixed > 0, sinon par snapshot
        const float amax = (amax_fixed > 0.0f) ? amax_fixed : max_abs(in, n);
        qmax  = (1L << (nbits - 1)) - 1;                       // 127 ou 32767
        scale = (amax > 0.0f) ? amax / static_cast<float>(qmax) : 1.0f;
        if (amax_used) *amax_used = amax;
    }

    const Header h = make_header(mode, nbits, n, payload, scale, step);

    out.resize(sizeof(Header) + payload);
    uint8_t* p = out.data() + sizeof(Header);

    switch (mode) {
        case Mode::Raw:
            if (n > 0) std::memcpy(p, in, payload);
            break;
        case Mode::F16:
            for (uint64_t i = 0; i < n; ++i) {
                const uint16_t v = float_to_half(in[i]);
                std::memcpy(p + 2 * i, &v, 2);
            }
            break;
        case Mode::IntN:
            for (uint64_t i = 0; i < n; ++i) {
                // Division en double : en float, 24 bits de mantisse ne suffisent
                // pas pour arrondir proprement jusqu'a 32767 (cas 16 bits).
                const double r = static_cast<double>(in[i]) / static_cast<double>(scale);
                long q = std::isfinite(r) ? std::lround(r) : 0;
                if (q > qmax || q < -qmax) ++clipped;  // sature (bornes globales)
                q = std::max(-qmax, std::min(qmax, q));
                if (nbits == 8) {
                    p[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
                } else {
                    const int16_t v = static_cast<int16_t>(q);
                    std::memcpy(p + 2 * i, &v, 2);
                }
            }
            break;
    }

    std::memcpy(out.data(), &h, sizeof h);  // en-tete ecrit en dernier
    if (n_clipped) *n_clipped = clipped;
    return true;
}

bool pack_intn(const void* q, uint64_t n, uint8_t nbits, float scale,
               uint32_t step, std::vector<uint8_t>& out) {
    out.clear();
    if (!valid_nbits(nbits)) return false;
    if (n > 0 && q == nullptr) return false;
    const uint64_t payload = n * (nbits / 8);
    const Header h = make_header(Mode::IntN, nbits, n, payload, scale, step);
    out.resize(sizeof(Header) + payload);
    if (n > 0) std::memcpy(out.data() + sizeof(Header), q, payload);
    std::memcpy(out.data(), &h, sizeof h);
    return true;
}

bool dequantize(const uint8_t* in, uint64_t nbytes, std::vector<float>& out) {
    out.clear();
    if (in == nullptr || nbytes < sizeof(Header)) return false;

    Header h;
    std::memcpy(&h, in, sizeof h);
    if (h.magic != MAGIC || h.version != VERSION) return false;
    if (h.flags != 0) return false;  // etapes sans perte a defaire avant

    uint64_t elem = 0;
    switch (h.mode) {
        case Mode::Raw: elem = 4; break;
        case Mode::F16: elem = 2; break;
        case Mode::IntN:
            if (!valid_nbits(h.nbits)) return false;
            if (!(h.scale > 0.0f) || !std::isfinite(h.scale)) return false;
            elem = h.nbits / 8;
            break;
        default: return false;
    }
    if (h.n_elements > nbytes) return false;  // evite un debordement du produit
    if (h.payload_size != h.n_elements * elem) return false;
    if (h.payload_size > nbytes - sizeof(Header)) return false;

    const uint8_t* p = in + sizeof(Header);
    const uint64_t n = h.n_elements;
    out.resize(n);

    switch (h.mode) {
        case Mode::Raw:
            if (n > 0) std::memcpy(out.data(), p, h.payload_size);
            break;
        case Mode::F16:
            for (uint64_t i = 0; i < n; ++i) {
                uint16_t v;
                std::memcpy(&v, p + 2 * i, 2);
                out[i] = half_to_float(v);
            }
            break;
        case Mode::IntN:
            for (uint64_t i = 0; i < n; ++i) {
                long q;
                if (h.nbits == 8) {
                    q = static_cast<int8_t>(p[i]);
                } else {
                    int16_t v;
                    std::memcpy(&v, p + 2 * i, 2);
                    q = v;
                }
                out[i] = static_cast<float>(q) * h.scale;
            }
            break;
    }
    return true;
}

ErrorStats compare(const float* ref, const float* rec, uint64_t n) {
    double sum2 = 0.0, maxerr = 0.0, amax = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(rec[i]) - static_cast<double>(ref[i]);
        sum2   += d * d;
        maxerr  = std::max(maxerr, std::fabs(d));
        amax    = std::max(amax, std::fabs(static_cast<double>(ref[i])));
    }
    if (n == 0 || amax == 0.0) return {0.0, 0.0};  // rien a normaliser
    return {std::sqrt(sum2 / static_cast<double>(n)) / amax, maxerr / amax};
}

double ratio(uint64_t original_bytes, uint64_t compressed_bytes) {
    if (compressed_bytes == 0) return 0.0;
    return static_cast<double>(original_bytes) / static_cast<double>(compressed_bytes);
}

}  // namespace cmp