// compression_config.hpp
#pragma once
#include <sstream>
#include <string>
#include "compression.hpp"

struct CompressConfig {
  bool      enabled = false;
  cmp::Mode mode    = cmp::Mode::Raw;
  uint8_t   nbits   = 0;
  bool      delta   = false;  
  bool      rle     = false;
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
    } else if (tok == "delta") { c.delta = true; }
    else if (tok == "rle")     { c.rle = true; }
    else { err = "etape inconnue: " + tok; return false; }
  }
  return true;
}