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

// ---------------------------------------------------------------------------
// DELTA stage: replace each element by its difference with the previous one
// (the first element is kept as is). Neighbouring nodes have close values, so
// the differences are small and repeat more often, which helps the RLE.
// The arithmetic is done on unsigned integers (modulo 2^bits), so it is
// exactly reversible, with no overflow problem.
// ---------------------------------------------------------------------------

// Width in bytes of one payload element, deduced from the header (0 if invalid).
inline std::size_t delta_width(const Header& h)
{
  switch (h.mode)
  {
    case Mode::Raw:  return 4;
    case Mode::F16:  return 2;
    case Mode::IntN: return h.nbits / 8;
  }
  return 0;
}

template <typename T>
inline void delta_forward(std::uint8_t* p, std::size_t count)
{
  T prev = 0;
  for (std::size_t i = 0; i < count; i++)
  {
    T cur;
    std::memcpy(&cur, p + i * sizeof(T), sizeof(T));
    const T d = static_cast<T>(cur - prev);
    std::memcpy(p + i * sizeof(T), &d, sizeof(T));
    prev = cur;
  }
}

template <typename T>
inline void delta_backward(std::uint8_t* p, std::size_t count)
{
  T prev = 0;
  for (std::size_t i = 0; i < count; i++)
  {
    T d;
    std::memcpy(&d, p + i * sizeof(T), sizeof(T));
    const T cur = static_cast<T>(d + prev);
    std::memcpy(p + i * sizeof(T), &cur, sizeof(T));
    prev = cur;
  }
}

// Apply DELTA in place on the payload of a buffer produced by quantize().
// Must be called BEFORE rle_apply(). Sets the DELTA flag; payload_size does not
// change. Returns false if the buffer is invalid, already DELTA- or RLE-coded.
inline bool delta_apply(std::vector<std::uint8_t>& buf)
{
  if (buf.size() < sizeof(Header)) return false;
  Header h;
  std::memcpy(&h, buf.data(), sizeof(Header));
  if (h.magic != MAGIC || h.version != VERSION) return false;
  if (h.flags & (DELTA | RLE)) return false;
  if (buf.size() != sizeof(Header) + h.payload_size) return false;

  const std::size_t w = delta_width(h);
  if (w != 1 && w != 2 && w != 4) return false;
  if (h.payload_size % w != 0) return false;
  const std::size_t count = static_cast<std::size_t>(h.payload_size) / w;

  std::uint8_t* p = buf.data() + sizeof(Header);
  if (w == 1) delta_forward<std::uint8_t>(p, count);
  else if (w == 2) delta_forward<std::uint16_t>(p, count);
  else delta_forward<std::uint32_t>(p, count);

  h.flags = static_cast<std::uint8_t>(h.flags | DELTA);
  std::memcpy(buf.data(), &h, sizeof(Header));
  return true;
}

// Inverse of delta_apply(). Must be called AFTER rle_undo() (the RLE flag must
// already be cleared). Clears the DELTA flag so the buffer can be given to
// dequantize(). Returns false if the buffer is invalid or not DELTA-coded.
inline bool delta_undo(std::vector<std::uint8_t>& buf)
{
  if (buf.size() < sizeof(Header)) return false;
  Header h;
  std::memcpy(&h, buf.data(), sizeof(Header));
  if (h.magic != MAGIC || h.version != VERSION) return false;
  if (!(h.flags & DELTA) || (h.flags & RLE)) return false;
  if (buf.size() != sizeof(Header) + h.payload_size) return false;

  const std::size_t w = delta_width(h);
  if (w != 1 && w != 2 && w != 4) return false;
  if (h.payload_size % w != 0) return false;
  const std::size_t count = static_cast<std::size_t>(h.payload_size) / w;

  std::uint8_t* p = buf.data() + sizeof(Header);
  if (w == 1) delta_backward<std::uint8_t>(p, count);
  else if (w == 2) delta_backward<std::uint16_t>(p, count);
  else delta_backward<std::uint32_t>(p, count);

  h.flags = static_cast<std::uint8_t>(h.flags & ~DELTA);
  std::memcpy(buf.data(), &h, sizeof(Header));
  return true;
}
}  // namespace cmp
