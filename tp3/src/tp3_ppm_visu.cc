#include "tp3_ppm_visu.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;

static double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static void appendNumber(std::string& target, unsigned value) {
    target += std::to_string(value);
}

static std::string makeP3(int width, int height, const std::vector<std::uint8_t>& rgb) {
    std::string content = "P3\n" + std::to_string(width) + " " +
                          std::to_string(height) + "\n255\n";
    content.reserve(content.size() + rgb.size() * 4);
    for (std::size_t i = 0; i < rgb.size(); i += 3) {
        appendNumber(content, rgb[i]);
        content.push_back(' ');
        appendNumber(content, rgb[i + 1]);
        content.push_back(' ');
        appendNumber(content, rgb[i + 2]);
        content.push_back('\n');
    }
    return content;
}

static std::string makeP6(int width, int height, const std::vector<std::uint8_t>& rgb) {
    std::string content = "P6\n" + std::to_string(width) + " " +
                          std::to_string(height) + "\n255\n";
    content.append(reinterpret_cast<const char*>(rgb.data()), rgb.size());
    return content;
}

static TP3PPMWriteMetrics writeFile(const std::string& path, const std::string& format,
                                    const std::string& content, bool sync_to_disk,
                                    double encode_ms, Clock::time_point initial) {
    TP3PPMWriteMetrics result;
    result.path = path;
    result.format = format;
    result.encode_ms = encode_ms;
    result.fsync_enabled = sync_to_disk;

    const auto start_write = Clock::now();
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Impossible d'ouvrir " + path + ": " + std::strerror(errno));
    }

    bool descriptor_open = true;
    try {
        std::size_t offset = 0;
        while (offset < content.size()) {
            const ssize_t written = ::write(fd, content.data() + offset, content.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                throw std::runtime_error("Echec write pour " + path + ": " + std::strerror(errno));
            }
            offset += static_cast<std::size_t>(written);
        }
        const auto written_at = Clock::now();
        result.write_ms = milliseconds(start_write, written_at);

        if (sync_to_disk) {
            if (::fsync(fd) != 0) {
                throw std::runtime_error("Echec fsync pour " + path + ": " + std::strerror(errno));
            }
            result.fsync_ms = milliseconds(written_at, Clock::now());
        }

        const int close_result = ::close(fd);
        descriptor_open = false;
        if (close_result != 0) {
            throw std::runtime_error("Echec close pour " + path + ": " + std::strerror(errno));
        }
    } catch (...) {
        if (descriptor_open) ::close(fd);
        throw;
    }

    result.size_bytes = std::filesystem::file_size(path);
    result.total_ms = milliseconds(initial, Clock::now());
    return result;
}
}

void validatePPM(const std::vector<std::uint8_t>& rgb, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Dimensions PPM invalides");
    }
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    if (w > std::numeric_limits<std::size_t>::max() / h ||
        w * h > std::numeric_limits<std::size_t>::max() / 3 ||
        rgb.size() != 3 * w * h) {
        throw std::invalid_argument("Taille du tampon RGB incorrecte");
    }
}

TP3PPMWriteMetrics writePPM_P3(const std::string& path, int width, int height,
                                const std::vector<std::uint8_t>& rgb, bool sync_to_disk) {
    validatePPM(rgb, width, height);
    const auto start = Clock::now();
    const auto content = makeP3(width, height, rgb);
    const auto encoded_at = Clock::now();
    return writeFile(path, "P3", content, sync_to_disk,
                     milliseconds(start, encoded_at), start);
}

TP3PPMWriteMetrics writePPM_P6(const std::string& path, int width, int height,
                                const std::vector<std::uint8_t>& rgb, bool sync_to_disk) {
    validatePPM(rgb, width, height);
    const auto start = Clock::now();
    const auto content = makeP6(width, height, rgb);
    const auto encoded_at = Clock::now();
    return writeFile(path, "P6", content, sync_to_disk,
                     milliseconds(start, encoded_at), start);
}
