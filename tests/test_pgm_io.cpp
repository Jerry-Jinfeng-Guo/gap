/**
 * @file test_pgm_io.cpp
 * @brief Test PGM JSON IO functionality
 */

#include <cassert>
#include <iostream>

#include "gap/core/types.h"
#include "gap/io/io_interface.h"

using namespace gap;
using namespace gap::io;

// Forward declare the JsonIOModule since it's internal
class JsonIOModule : public IIOModule {
  public:
    NetworkData read_network_data(const std::string& filename) override;
    void write_results(const std::string& filename, const ComplexVector& bus_voltages,
                       bool converged, int iterations) override;
    bool validate_input_format(const std::string& filename) override;
};

void test_pgm_json_io() {
    std::cout << "Testing PGM JSON IO with network_1.json..." << std::endl;

    JsonIOModule io;

    try {
        // Test reading the PGM input file
        NetworkData network = io.read_network_data("data/pgm/network_1.json");

        // Verify structure
        std::cout << "Network structure:" << std::endl;
        std::cout << "  Version: " << network.version << std::endl;
        std::cout << "  Type: " << network.type << std::endl;
        std::cout << "  Is batch: " << (network.is_batch ? "true" : "false") << std::endl;
        std::cout << "  Buses: " << network.num_buses << std::endl;
        std::cout << "  Branches: " << network.num_branches << std::endl;
        std::cout << "  Appliances: " << network.num_appliances << std::endl;

        // Verify expected counts from network_1.json
        assert(network.num_buses == 4);       // nodes 1,2,3,4
        assert(network.num_branches == 3);    // lines 5,6,7
        assert(network.num_appliances == 2);  // source 8, load 9

        // Verify bus data
        std::cout << "\nBus details:" << std::endl;
        for (const auto& bus : network.buses) {
            std::cout << "  Bus " << bus.id << ": u_rated=" << bus.u_rated
                      << "V, type=" << static_cast<int>(bus.bus_type) << std::endl;
            assert(bus.u_rated == 400.0);  // All nodes have u_rated=400V
        }

        // Verify branch data
        std::cout << "\nBranch details:" << std::endl;
        for (const auto& branch : network.branches) {
            std::cout << "  Branch " << branch.id << ": " << branch.from_bus << "->"
                      << branch.to_bus << ", r1=" << branch.r1 << ", x1=" << branch.x1
                      << ", i_n=" << branch.i_n << std::endl;
            assert(branch.r1 == 0.206e-1);  // All lines have same parameters
            assert(branch.x1 == 0.079e-1);
            assert(branch.i_n == 1e3);
        }

        // Verify appliance data
        std::cout << "\nAppliance details:" << std::endl;
        for (const auto& appliance : network.appliances) {
            std::cout << "  Appliance " << appliance.id << " at bus " << appliance.node
                      << ", type=" << static_cast<int>(appliance.type);
            if (appliance.type == ApplianceType::SOURCE) {
                std::cout << ", u_ref=" << appliance.u_ref;
            } else if (appliance.type == ApplianceType::LOADGEN) {
                std::cout << ", p_spec=" << appliance.p_specified
                          << ", q_spec=" << appliance.q_specified;
            }
            std::cout << std::endl;
        }

        // Verify bus types were correctly inferred
        bool found_slack = false;
        for (const auto& bus : network.buses) {
            if (bus.bus_type == BusType::SLACK) {
                found_slack = true;
                assert(bus.id == 1);  // Bus 1 should be slack (has source)
            }
        }
        assert(found_slack);

        // Verify PGM compliance
        assert(network.validate_pgm_compliance());

        std::cout << "\n✓ PGM JSON IO test passed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(1);
    }
}

int main() {
    test_pgm_json_io();
    return 0;
}