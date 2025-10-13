#pragma once

#include <memory>
#include <string>

#include "gap/core/types.h"

namespace gap::io {

/**
 * @brief Abstract interface for input/output operations
 */
class IIOModule {
  public:
    virtual ~IIOModule() = default;

    /**
     * @brief Read network data from a file
     * @param filename Path to the input file
     * @return NetworkData structure containing the power system data
     */
    virtual NetworkData read_network_data(const std::string& filename) = 0;

    /**
     * @brief Write results to a file
     * @param filename Path to the output file
     * @param bus_voltages Vector of bus voltage phasors
     * @param converged Whether the power flow converged
     * @param iterations Number of iterations taken
     */
    virtual void write_results(const std::string& filename, const ComplexVector& bus_voltages,
                               bool converged, int iterations) = 0;

    /**
     * @brief Validate input data format
     * @param filename Path to the file to validate
     * @return true if file format is valid
     */
    virtual bool validate_input_format(const std::string& filename) = 0;
};

}  // namespace gap::io
