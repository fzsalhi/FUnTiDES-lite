#pragma once

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tp3_ppm_visu.hpp"

struct TP3FrameMetrics {
    double extract_ms = 0.0;
    double color_ms = 0.0;
    double total_ms = 0.0;
    std::vector<TP3PPMWriteMetrics> files;
};

class TP3SliceCPU {
public:
    template <class Mesh>
    TP3SliceCPU(Mesh& mesh, const int elements[3], const float lengths[3],
                const std::string& plane, float position, int width, int height)
        : TP3SliceCPU(
              mesh.getOrder(), elements, lengths, plane, position, width, height,
              [&mesh](int element, int i, int j, int k) {
                  return mesh.globalNodeIndex(element, i, j, k);
              },
              [&mesh](int node, int dimension) {
                  return static_cast<float>(mesh.nodeCoord(node, dimension));
              }) {}

    template <class PressureView>
    std::vector<float> extract(const PressureView& pressure, int column) const {
        if (column < 0 || static_cast<std::size_t>(column) >= pressure.extent(1)) {
            throw std::out_of_range("Colonne de pression invalide");
        }

        const int count = width_ * height_;
        Kokkos::View<float*, Kokkos::HostSpace> samples("tp3_slice_samples", count);
        auto indices = node_indices_;
        auto source = pressure;

        Kokkos::parallel_for(
            "tp3_slice_extract_cpu", Kokkos::RangePolicy<Kokkos::OpenMP>(0, count),
            KOKKOS_LAMBDA(const int index) {
                samples(index) = static_cast<float>(source(indices(index), column));
            });
        Kokkos::fence();

        std::vector<float> values(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            values[static_cast<std::size_t>(i)] = samples(i);
        }
        return values;
    }

    std::vector<std::uint8_t> toRGB(const std::vector<float>& values,
                                    float fixed_scale = 0.0f) const;

    template <class PressureView>
    TP3FrameMetrics savePPM(const PressureView& pressure, int column,
                            const std::string& prefix,
                            const std::string& format = "both",
                            float fixed_scale = 0.0f,
                            bool sync_to_disk = false) const {
        if (format != "P3" && format != "P6" && format != "both") {
            throw std::invalid_argument("Format PPM invalide");
        }
        using Clock = std::chrono::steady_clock;
        TP3FrameMetrics metrics;
        Kokkos::fence();
        const auto start = Clock::now();
        const auto values = extract(pressure, column);
        const auto extracted = Clock::now();
        const auto rgb = toRGB(values, fixed_scale);
        const auto colored = Clock::now();
        metrics.extract_ms = std::chrono::duration<double, std::milli>(extracted - start).count();
        metrics.color_ms = std::chrono::duration<double, std::milli>(colored - extracted).count();
        if (format == "P3" || format == "both") {
            metrics.files.push_back(writePPM_P3(prefix + "_P3.ppm", width_, height_, rgb,
                                                sync_to_disk));
        }
        if (format == "P6" || format == "both") {
            metrics.files.push_back(writePPM_P6(prefix + "_P6.ppm", width_, height_, rgb,
                                                sync_to_disk));
        }
        metrics.total_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return metrics;
    }

    int width() const;
    int height() const;

private:
    using NodeIndex = std::function<int(int, int, int, int)>;
    using NodeCoordinate = std::function<float(int, int)>;

    TP3SliceCPU(int order, const int elements[3], const float lengths[3],
                const std::string& plane, float position, int width, int height,
                const NodeIndex& node_index, const NodeCoordinate& node_coordinate);

    static int checkedPixelCount(int width, int height);

    int width_;
    int height_;
    Kokkos::View<int*, Kokkos::HostSpace> node_indices_;
};
