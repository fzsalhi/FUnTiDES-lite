// compression.hpp
// Interface de compression avec perte pour les snapshots.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cmp {

constexpr uint32_t MAGIC   = 0x43504D31;  // identifiant du format (arbitraire)
constexpr uint16_t VERSION = 1;

enum class Mode : uint8_t { Raw = 0, F16 = 1, IntN = 2 };

enum Flags : uint8_t { DELTA = 1 << 0, RLE = 1 << 1 };

struct Header {
    uint32_t magic;
    uint16_t version;
    Mode     mode;         // Raw, F16 ou IntN
    uint8_t  nbits;        // 8 or 16 for IntN,else 0
    uint8_t  flags;        // DELTA, RLE
    uint8_t  reserved[7];
    uint64_t n_elements;   // number of floats before compression
    uint64_t payload_size;
    float    scale;        // scale
    uint32_t step;         // snapshot time step
};

// Garde-fous : le format doit être identique pour tout le groupe.
static_assert(std::is_trivially_copyable<Header>::value,
              "Header doit etre copiable par memcpy");
static_assert(sizeof(Header) == 40, "taille de l'en-tete inattendue");
static_assert(sizeof(Header) ==
                  sizeof(uint32_t) + sizeof(uint16_t) + sizeof(Mode) +
                  2 * sizeof(uint8_t) + sizeof(Header::reserved) +
                  2 * sizeof(uint64_t) + sizeof(float) + sizeof(uint32_t),
              "Header contient du remplissage cache");
static_assert(offsetof(Header, n_elements) == 16, "decalage inattendu");

// float -> octets (en-tete + donnees quantifiees). `out` est vide au debut.
// Pour mode == IntN, nbits doit valoir 8 ou 16 (ignore pour Raw et F16).
// Renvoie false si les parametres sont invalides (out reste vide).
//
// Bornes (mode IntN uniquement) :
//   amax_fixed == 0  -> bornes PAR SNAPSHOT : amax = max|p| du tableau
//   amax_fixed  > 0  -> bornes GLOBALES : amax fixe a l'avance ; les valeurs
//                       |p| > amax sont saturees (ecretees a +-qmax)
// Sorties optionnelles :
//   n_clipped : nombre de valeurs saturees (|q| > qmax avant ecretage)
//   amax_used : borne effectivement utilisee
bool quantize(const float* in, uint64_t n, Mode mode, uint8_t nbits,
              uint32_t step, std::vector<uint8_t>& out,
              float amax_fixed = 0.0f, uint64_t* n_clipped = nullptr,
              float* amax_used = nullptr);

// Assemble en-tete + payload a partir d'entiers DEJA quantifies (int8_t[n] ou
// int16_t[n], ordre natif). Utilise par la quantification GPU : l'en-tete est
// construit par le meme code que quantize(), donc identique octet par octet.
bool pack_intn(const void* q, uint64_t n, uint8_t nbits, float scale,
               uint32_t step, std::vector<uint8_t>& out);

// octets -> float (lit l'en-tete pour savoir quoi faire).
// Renvoie false si le tampon est invalide (magic, version, tailles,
// incoherence du mode) ou si des etapes sans perte (flags != 0) n'ont pas
// ete defaites avant : l'appelant doit d'abord decoder DELTA / RLE.
bool dequantize(const uint8_t* in, uint64_t nbytes, std::vector<float>& out);

// Mesures d'erreur, normalisees par max|p| (pas d'erreur relative point a point).
struct ErrorStats { double rmse_norm; double max_abs_norm; };

// Plus grande valeur absolue du tableau (0 si n == 0).
float max_abs(const float* x, uint64_t n);

// Compare ref (original) et rec (reconstruit), n elements chacun.
ErrorStats compare(const float* ref, const float* rec, uint64_t n);

// Octets originaux / octets compresses (> 1 si ca compresse, 0 si compresse == 0).
double ratio(uint64_t original_bytes, uint64_t compressed_bytes);

}  // namespace cmp