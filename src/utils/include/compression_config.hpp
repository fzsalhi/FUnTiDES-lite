// compression_config.hpp
#pragma once
#include <cstdint>
#include <sstream>
#include <string>
#include "compression.hpp"

// Syntaxe de --insitu-compress : <mode>[+etape]...
//   mode  : raw | f16 | int8 | int16
//   etapes: delta | rle                (pas encore disponibles)
//           gpu           quantification sur le GPU (int8/int16 seulement)
//           global=<amax> bornes globales : amax fixe a l'avance (defaut :
//                         bornes par snapshot)
//           check         (avec gpu) compare la sortie GPU a la version CPU,
//                         octet par octet
//           novalid       desactive dequantize + compare (mesure de cout)
// Exemples : int8   int16+gpu   int8+gpu+global=0.26+check   int16+novalid
struct CompressConfig {
  bool      enabled = false;
  cmp::Mode mode    = cmp::Mode::Raw;
  uint8_t   nbits   = 0;
  bool      delta   = false;
  bool      rle     = false;
  bool      gpu           = false;  // quantification sur le GPU
  bool      check         = false;  // verification octet par octet GPU vs CPU
  bool      validate      = true;   // dequantize + compare (erreur)
  bool      global_bounds = false;  // bornes globales (sinon par snapshot)
  float     amax          = 0.0f;   // borne globale (si global_bounds)
};

// Renvoie false et remplit err si la chaine est invalide.
inline bool parse_compress(const std::string& s, CompressConfig& c, std::string& err)
{
  c = CompressConfig{};
  if (s == "none" || s.empty()) return true;
  std::stringstream ss(s);
  std::string tok;
  bool first = true;
  while (std::getline(ss, tok, '+')) {
    if (first) {
      first = false;
      if      (tok == "raw")   { c.mode = cmp::Mode::Raw; }
      else if (tok == "f16")   { c.mode = cmp::Mode::F16; }
      else if (tok == "int8")  { c.mode = cmp::Mode::IntN; c.nbits = 8; }
      else if (tok == "int16") { c.mode = cmp::Mode::IntN; c.nbits = 16; }
      else { err = "mode de compression inconnu: " + tok; return false; }
      c.enabled = true;
    } else if (tok == "delta")   { c.delta = true; }
    else if (tok == "rle")       { c.rle = true; }
    else if (tok == "gpu")       { c.gpu = true; }
    else if (tok == "check")     { c.check = true; }
    else if (tok == "novalid")   { c.validate = false; }
    else if (tok.rfind("global=", 0) == 0) {
      try { c.amax = std::stof(tok.substr(7)); }
      catch (...) { err = "valeur de amax invalide: " + tok; return false; }
      if (!(c.amax > 0.0f)) { err = "global=<amax> exige amax > 0: " + tok; return false; }
      c.global_bounds = true;
    }
    else { err = "etape inconnue: " + tok; return false; }
  }
  if (c.global_bounds && c.mode != cmp::Mode::IntN) {
    err = "global=<amax> ne concerne que int8 et int16";
    return false;
  }
  if (c.gpu && c.mode != cmp::Mode::IntN) {
    err = "gpu ne concerne que int8 et int16 (f16/raw : version CPU)";
    return false;
  }
  if (c.check && !c.gpu) {
    err = "check n'a de sens qu'avec gpu";
    return false;
  }
  return true;
}