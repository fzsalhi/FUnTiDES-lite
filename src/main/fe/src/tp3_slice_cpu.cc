#include "tp3_slice_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

int TP3SliceCPU::checkedPixelCount(int width, int height) {
    if (width < 2 || height < 2 ||
        width > std::numeric_limits<int>::max() / height ||
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 3)) {
        throw std::invalid_argument("Resolution de la slice invalide");
    }
    return width * height;
}

TP3SliceCPU::TP3SliceCPU(int order, const int elements[3], const float lengths[3],
                         const std::string& plane, float position, int width, int height,
                         const NodeIndex& node_index,
                         const NodeCoordinate& node_coordinate)
    : width_(width),
      height_(height),
      node_indices_("tp3_slice_node_indices", checkedPixelCount(width, height)) {
    if (!std::isfinite(position) || position < 0.0f || position > 1.0f) {
        throw std::invalid_argument("Position de slice hors de [0, 1]");
    }
    if (order < 1) {
        throw std::invalid_argument("Ordre GLL invalide");
    }
    for (int d = 0; d < 3; ++d) {
        if (elements[d] <= 0 || !std::isfinite(lengths[d]) || lengths[d] <= 0.0f) {
            throw std::invalid_argument("Maillage invalide");
        }
    }

    int horizontal = 0;
    int vertical = 1;
    int fixed = 2;
    if (plane == "XZ") {
        horizontal = 0;
        vertical = 2;
        fixed = 1;
    } else if (plane == "YZ") {
        horizontal = 1;
        vertical = 2;
        fixed = 0;
    } else if (plane != "XY") {
        throw std::invalid_argument("Plan de coupe invalide");
    }

    for (int py = 0; py < height_; ++py) {
        for (int px = 0; px < width_; ++px) {
            float point[3] = {0.0f, 0.0f, 0.0f};
            point[horizontal] = lengths[horizontal] * static_cast<float>(px) / (width_ - 1);
            point[vertical] = lengths[vertical] * static_cast<float>(height_ - 1 - py) / (height_ - 1);
            point[fixed] = lengths[fixed] * position;

            int cell[3];
            for (int d = 0; d < 3; ++d) {
                const int index = static_cast<int>(point[d] / lengths[d] * elements[d]);
                cell[d] = std::max(0, std::min(elements[d] - 1, index));
            }

            const int element = cell[0] + elements[0] * (cell[1] + elements[1] * cell[2]);
            float min_distance = std::numeric_limits<float>::max();
            int nearest_node = -1;

            for (int k = 0; k <= order; ++k) {
                for (int j = 0; j <= order; ++j) {
                    for (int i = 0; i <= order; ++i) {
                        const int node = node_index(element, i, j, k);
                        const float dx = point[0] - node_coordinate(node, 0);
                        const float dy = point[1] - node_coordinate(node, 1);
                        const float dz = point[2] - node_coordinate(node, 2);
                        const float distance = dx * dx + dy * dy + dz * dz;
                        if (distance < min_distance) {
                            min_distance = distance;
                            nearest_node = node;
                        }
                    }
                }
            }

            if (nearest_node < 0) {
                throw std::runtime_error("Aucun noeud GLL trouve");
            }
            node_indices_(py * width_ + px) = nearest_node;
        }
    }
}

std::vector<std::uint8_t> TP3SliceCPU::toRGB(const std::vector<float>& values,
                                             float fixed_scale) const {
    const int count = width_ * height_;
    if (values.size() != static_cast<std::size_t>(count)) {
        throw std::invalid_argument("Taille de slice incorrecte");
    }
    if (!std::isfinite(fixed_scale) || fixed_scale < 0.0f) {
        throw std::invalid_argument("Echelle de normalisation invalide");
    }

    Kokkos::View<float*, Kokkos::HostSpace> samples("tp3_slice_values", count);
    Kokkos::View<std::uint8_t*, Kokkos::HostSpace> pixels("tp3_slice_rgb", 3 * count);

    for (int i = 0; i < count; ++i) {
        const float value = values[static_cast<std::size_t>(i)];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("La slice contient une valeur non finie");
        }
        samples(i) = value;
    }

    float max_abs = 0.0f;
    Kokkos::parallel_reduce(
        "tp3_slice_max_abs_cpu", Kokkos::RangePolicy<Kokkos::OpenMP>(0, count),
        KOKKOS_LAMBDA(const int index, float& local_max) {
            const float x = samples(index);
            const float magnitude = x < 0.0f ? -x : x;
            if (magnitude > local_max) {
                local_max = magnitude;
            }
        },
        Kokkos::Max<float>(max_abs));

    const float scale = fixed_scale > 0.0f ? fixed_scale : (max_abs > 0.0f ? max_abs : 1.0f);
    Kokkos::parallel_for(
        "tp3_slice_rgb_cpu", Kokkos::RangePolicy<Kokkos::OpenMP>(0, count),
        KOKKOS_LAMBDA(const int index) {
            float normalized = samples(index) / scale;
            normalized = normalized < -1.0f ? -1.0f : (normalized > 1.0f ? 1.0f : normalized);
            const float magnitude = normalized < 0.0f ? -normalized : normalized;
            const std::uint8_t pale = static_cast<std::uint8_t>(255.0f * (1.0f - magnitude) + 0.5f);
            pixels(3 * index) = normalized < 0.0f ? pale : 255;
            pixels(3 * index + 1) = pale;
            pixels(3 * index + 2) = normalized > 0.0f ? pale : 255;
        });
    Kokkos::fence();

    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(count) * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = pixels(static_cast<int>(i));
    }
    return rgb;
}

int TP3SliceCPU::width() const {
    return width_;
}

int TP3SliceCPU::height() const {
    return height_;
}
