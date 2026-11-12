#pragma once
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace SliceUtils
{
struct Plane
{
  int axis;
  float coordinate;
  std::vector<int> nodes;
};
struct PixelGrid
{
  float coordinate;
  std::vector<int> nodes;
};
inline int fixedAxis(const std::string& plane)
{
  if (plane == "XY") return 2;
  if (plane == "XZ") return 1;
  if (plane == "YZ") return 0;
  throw std::invalid_argument("Plan de coupe invalide");
}
template <class Mesh>
Plane selectPlane(Mesh& mesh, const std::string& name, float position)
{
  const int axis = fixedAxis(name), count = mesh.getNumberOfNodes();
  if (count <= 0 || !std::isfinite(position))
    throw std::invalid_argument("Plan ou maillage invalide");
  Plane result{axis, float(mesh.nodeCoord(0, axis)), {}};
  for (int n = 1; n < count; ++n)
  {
    const float x = mesh.nodeCoord(n, axis), d = std::abs(x - position),
                best = std::abs(result.coordinate - position);
    if (d < best || (d == best && x < result.coordinate)) result.coordinate = x;
  }
  for (int n = 0; n < count; ++n)
    if (mesh.nodeCoord(n, axis) == result.coordinate) result.nodes.push_back(n);
  return result;
}
inline float nearest(const std::vector<float>& levels, float x)
{
  auto it = std::lower_bound(levels.begin(), levels.end(), x);
  if (it == levels.begin()) return *it;
  if (it == levels.end()) return levels.back();
  const float high = *it, low = *(it - 1);
  return x - low <= high - x ? low : high;
}
template <class Mesh>
PixelGrid uniformPixels(Mesh& mesh, const float lengths[3],
                        const std::string& name, float relative, int width,
                        int height)
{
  if (width < 2 || height < 2 ||
      width > std::numeric_limits<int>::max() / height / 3 ||
      !std::isfinite(relative) || relative < 0 || relative > 1)
    throw std::invalid_argument("Grille de pixels invalide");
  const int axis = fixedAxis(name), u = axis == 0 ? 1 : 0,
            v = axis == 2 ? 1 : 2;
  const auto plane = selectPlane(mesh, name, relative * lengths[axis]);
  std::vector<float> xs, ys;
  std::map<std::pair<float, float>, int> nodes;
  for (int n : plane.nodes)
  {
    const float x = mesh.nodeCoord(n, u), y = mesh.nodeCoord(n, v);
    xs.push_back(x);
    ys.push_back(y);
    nodes[{x, y}] = n;
  }
  auto unique = [](auto& a) {
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
  };
  unique(xs);
  unique(ys);
  if (xs.size() * ys.size() != nodes.size())
    throw std::runtime_error("Le plan doit former une grille cartesienne");
  PixelGrid result{plane.coordinate,
                   std::vector<int>(std::size_t(width) * height)};
  for (int py = 0; py < height; ++py)
    for (int px = 0; px < width; ++px)
    {
      const float x = lengths[u] * float(px) / (width - 1),
                  y = lengths[v] * float(height - 1 - py) / (height - 1);
      result.nodes[py * width + px] =
          nodes.at({nearest(xs, x), nearest(ys, y)});
    }
  return result;
}
template <class ExecutionSpace, class Pressure, class Indices, class Values>
void gather(const Pressure& pressure, int column, const Indices& nodes,
            const Values& values)
{
  Kokkos::parallel_for(
      "gather_slice", Kokkos::RangePolicy<ExecutionSpace>(0, nodes.extent(0)),
      KOKKOS_LAMBDA(int n) { values(n) = pressure(nodes(n), column); });
}
}  // namespace SliceUtils
