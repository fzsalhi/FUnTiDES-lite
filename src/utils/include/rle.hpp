#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cstring>
#include "compression.hpp"

// Lossless run-length encoding, as a stage of the compression pipeline.
// Format: a sequence of pairs (count, value). `count` is one byte in
// [1, 255]; longer runs are split into several pairs.
namespace cmp
{
// Encode a byte buffer. Worst case (no repeated byte): output is twice as
// large as the input.
inline std::vector<std::uint8_t> rle_encode_bytes(
    const std::vector<std::uint8_t>& in)
{
  std::vector<std::uint8_t> out;
  if (in.empty()) return out;

  std::uint8_t value = in[0];  // byte of the current run
  unsigned count = 1;          // length of the current run

  for (std::size_t i = 1; i < in.size(); i++)
  {
    if (in[i] == value && count < 255)
    {
      count++;
    }
    else
    {
      // close the current run and start a new one
      out.push_back(static_cast<std::uint8_t>(count));
      out.push_back(value);
      value = in[i];
      count = 1;
    }
  }
  // do not forget the last run
  out.push_back(static_cast<std::uint8_t>(count));
  out.push_back(value);
  return out;
}

// Decode a buffer produced by rle_encode_bytes(). Throws on an invalid
// buffer (odd size, or a count equal to zero).
inline std::vector<std::uint8_t> rle_decode_bytes(
    const std::vector<std::uint8_t>& in)
{
  if (in.size() % 2 != 0)
    throw std::runtime_error("rle_decode_bytes: odd buffer size");

  std::vector<std::uint8_t> out;
  for (std::size_t i = 0; i < in.size(); i += 2)
  {
    const std::uint8_t count = in[i];
    const std::uint8_t value = in[i + 1];
    if (count == 0)
      throw std::runtime_error("rle_decode_bytes: run of length 0");
    out.insert(out.end(), count, value);
  }
  return out;
}

// Apply RLE to the payload of a buffer produced by quantize() (header +
// payload). The header stays readable: the RLE flag is set and payload_size
// becomes the size of the compressed payload. Returns false if the buffer is
// invalid or if RLE has already been applied.
// Pipeline order: quantize -> (DELTA) -> RLE; undo in the reverse order.
inline bool rle_apply(std::vector<std::uint8_t>& buf)
{
  if (buf.size() < sizeof(Header)) return false;
  Header h;
  std::memcpy(&h, buf.data(), sizeof(Header));
  if (h.magic != MAGIC || h.version != VERSION) return false;
  if (h.flags & RLE) return false;
  if (buf.size() != sizeof(Header) + h.payload_size) return false;

  const std::vector<std::uint8_t> payload(buf.begin() + sizeof(Header),
                                          buf.end());
  const std::vector<std::uint8_t> enc = rle_encode_bytes(payload);

  h.flags = static_cast<std::uint8_t>(h.flags | RLE);
  h.payload_size = enc.size();

  buf.resize(sizeof(Header));
  buf.insert(buf.end(), enc.begin(), enc.end());
  std::memcpy(buf.data(), &h, sizeof(Header));
  return true;
}

// Inverse of rle_apply(): restore the uncompressed payload and clear the RLE
// flag, so that the buffer can then be given to dequantize() (or to the
// DELTA decoder). Returns false if the buffer is invalid or not RLE-coded.
inline bool rle_undo(std::vector<std::uint8_t>& buf)
{
  if (buf.size() < sizeof(Header)) return false;
  Header h;
  std::memcpy(&h, buf.data(), sizeof(Header));
  if (h.magic != MAGIC || h.version != VERSION) return false;
  if (!(h.flags & RLE)) return false;
  if (buf.size() != sizeof(Header) + h.payload_size) return false;

  const std::vector<std::uint8_t> enc(buf.begin() + sizeof(Header),
                                      buf.end());
  std::vector<std::uint8_t> payload;
  try
  {
    payload = rle_decode_bytes(enc);
  }
  catch (const std::runtime_error&)
  {
    return false;
  }

  h.flags = static_cast<std::uint8_t>(h.flags & ~RLE);
  h.payload_size = payload.size();

  buf.resize(sizeof(Header));
  buf.insert(buf.end(), payload.begin(), payload.end());
  std::memcpy(buf.data(), &h, sizeof(Header));
  return true;
}
}  // namespace cmp
