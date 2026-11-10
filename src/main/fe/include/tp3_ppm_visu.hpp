#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct TP3PPMWriteMetrics {
    std::string format;
    std::string path;
    std::size_t size_bytes = 0;
    double encode_ms = 0.0;
    double write_ms = 0.0;
    double fsync_ms = 0.0;
    double total_ms = 0.0;
    bool fsync_enabled = false;
};

void validatePPM(const std::vector<std::uint8_t>& rgb, int width, int height);

TP3PPMWriteMetrics writePPM_P3(const std::string& path, int width, int height,
                                const std::vector<std::uint8_t>& rgb,
                                bool sync_to_disk = false);

TP3PPMWriteMetrics writePPM_P6(const std::string& path, int width, int height,
                                const std::vector<std::uint8_t>& rgb,
                                bool sync_to_disk = false);
