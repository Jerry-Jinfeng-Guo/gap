/**
 * @file test_pgm_io.cpp
 * @brief Test PGM JSON IO functionality
 */

#include <cassert>
#include <filesystem>
#include <iostream>

#include "gap/core/backend_factory.h"
#include "gap/core/types.h"
#include "gap/io/io_interface.h"

#include "test_framework.h"

using namespace gap;

// Helper function to find the correct path to test data files
std::string find_data_file(const std::string& relative_path) {
    // Get current working directory for path resolution
    std::string cwd = std::filesystem::current_path().string();

    // List of possible paths to try in order
    std::vector<std::string> candidates = {
        // Direct path from project root
        relative_path,
        // From build directory
        "../" + relative_path,
        // From build/bin directory (VS Code Test Explorer often runs from here)
        "../../" + relative_path,
        // From build/tests directory
        "../../" + relative_path,
        // From any subdirectory in build/
        "../" + relative_path,
        // Absolute construction based on current directory
        cwd + "/" + relative_path};

    // If we can detect we're in a build-related directory, add more candidates
    if (cwd.find("/build") != std::string::npos) {
        // Find the project root by going up from build directory
        std::string project_root = cwd;
        size_t build_pos = project_root.find("/build");
        if (build_pos != std::string::npos) {
            project_root = project_root.substr(0, build_pos);
            candidates.push_back(project_root + "/" + relative_path);
        }
    }

    // Try each candidate path
    for (auto const& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    // Last resort: return original path and let it fail with clear error
    std::cerr << "ERROR: Could not find " << relative_path << " in any of " << candidates.size()
              << " expected locations from working directory: " << cwd << std::endl;
    return relative_path;
}

using namespace gap::io;
using namespace gap::core;

void test_pgm_json_io() {
    std::cout << "Testing PGM JSON IO with network_1.json..." << std::endl;

    auto io = BackendFactory::create_io_module();

    try {
        // Test reading the PGM input file
        std::string network_file = find_data_file("data/pgm/network_1.json");
        NetworkData network = io->read_network_data(network_file);

        // Verify structure
        std::cout << "Network structure:" << std::endl;
        std::cout << "  Version: " << network.version << std::endl;
        std::cout << "  Type: " << network.type << std::endl;
        std::cout << "  Is batch: " << (network.is_batch ? "true" : "false") << std::endl;
        std::cout << "  Buses: " << network.num_buses << std::endl;
        std::cout << "  Branches: " << network.num_branches << std::endl;
        std::cout << "  Appliances: " << network.num_appliances << std::endl;

        // Verify expected counts from network_1.json
        ASSERT_EQ(4, network.num_buses);       // nodes 1,2,3,4
        ASSERT_EQ(3, network.num_branches);    // lines 5,6,7
        ASSERT_EQ(2, network.num_appliances);  // source 8, load 9

        // Verify bus data
        std::cout << "\nBus details:" << std::endl;
        for (auto const& bus : network.buses) {
            std::cout << "  Bus " << bus.id << ": u_rated=" << bus.u_rated
                      << "V, type=" << static_cast<int>(bus.bus_type) << std::endl;
            ASSERT_NEAR(400.0, bus.u_rated, 1e-6);  // All nodes have u_rated=400V
        }

        // Verify branch data
        std::cout << "\nBranch details:" << std::endl;
        for (auto const& branch : network.branches) {
            std::cout << "  Branch " << branch.id << ": " << branch.from_bus << "->"
                      << branch.to_bus << ", r1=" << branch.r1 << ", x1=" << branch.x1
                      << ", i_n=" << branch.i_n << std::endl;
            ASSERT_NEAR(0.206e-1, branch.r1, 1e-6);  // All lines have same parameters
            ASSERT_NEAR(0.079e-1, branch.x1, 1e-6);
            ASSERT_NEAR(1e3, branch.i_n, 1e-6);
        }

        // Verify appliance data
        std::cout << "\nAppliance details:" << std::endl;
        for (auto const& appliance : network.appliances) {
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
        for (auto const& bus : network.buses) {
            if (bus.bus_type == BusType::SLACK) {
                found_slack = true;
                ASSERT_EQ(1, bus.id);  // Bus 1 should be slack (has source)
            }
        }
        ASSERT_TRUE(found_slack);

        // Verify PGM compliance
        ASSERT_TRUE(network.validate_pgm_compliance());

        std::cout << "\n✓ PGM JSON IO test passed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(1);
    }
}

void test_transformer_network() {
    std::cout << "\nTesting Transformer Network (network_2.json)..." << std::endl;

    auto io = BackendFactory::create_io_module();

    try {
        std::string network_file = find_data_file("data/pgm/network_2.json");
        NetworkData network = io->read_network_data(network_file);

        // Verify structure for transformer network
        std::cout << "Transformer network structure:" << std::endl;
        std::cout << "  Buses: " << network.num_buses << std::endl;
        std::cout << "  Branches: " << network.num_branches << std::endl;
        std::cout << "  Appliances: " << network.num_appliances << std::endl;

        // Verify expected counts from network_2.json
        ASSERT_EQ(2, network.num_buses);       // nodes 1,2
        ASSERT_EQ(1, network.num_branches);    // transformer 3
        ASSERT_EQ(1, network.num_appliances);  // source 4

        // Verify bus voltage ratings (different for transformer network)
        std::cout << "\nTransformer bus details:" << std::endl;
        for (auto const& bus : network.buses) {
            std::cout << "  Bus " << bus.id << ": u_rated=" << bus.u_rated << "V" << std::endl;
        }

        // Check that we have the expected voltage levels
        bool found_10kv = false, found_400v = false;
        for (auto const& bus : network.buses) {
            if (std::abs(bus.u_rated - 10000.0) < 1e-6) found_10kv = true;
            if (std::abs(bus.u_rated - 400.0) < 1e-6) found_400v = true;
        }
        ASSERT_TRUE(found_10kv && found_400v);  // Should have both voltage levels

        // Verify transformer branch data
        std::cout << "\nTransformer branch details:" << std::endl;
        auto const& transformer = network.branches[0];
        std::cout << "  Transformer " << transformer.id << ": " << transformer.from_bus << "->"
                  << transformer.to_bus << std::endl;
        std::cout << "    r1=" << transformer.r1 << ", x1=" << transformer.x1 << std::endl;
        std::cout << "    b1=" << transformer.b1 << ", g1=" << transformer.g1 << std::endl;
        std::cout << "    i_n=" << transformer.i_n
                  << ", type=" << static_cast<int>(transformer.branch_type) << std::endl;

        // Verify transformer was properly converted from PGM parameters
        ASSERT_TRUE(transformer.branch_type == BranchType::TRAFO);
        ASSERT_TRUE(transformer.r1 > 0.0);   // Should have calculated resistance from uk/pk
        ASSERT_TRUE(transformer.x1 > 0.0);   // Should have calculated reactance
        ASSERT_TRUE(transformer.i_n > 0.0);  // Should have rated current from sn/u_rated

        // Verify appliance (source) data
        auto const& source = network.appliances[0];
        ASSERT_TRUE(source.type == ApplianceType::SOURCE);
        ASSERT_NEAR(1.0, source.u_ref, 1e-6);  // Reference voltage

        std::cout << "\n✓ Transformer network test passed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Transformer test error: " << e.what() << std::endl;
        std::exit(1);
    }
}

void test_generic_branch_network() {
    std::cout << "\nTesting Generic Branch Network (network_3.json)..." << std::endl;

    auto io = BackendFactory::create_io_module();

    try {
        std::string network_file = find_data_file("data/pgm/network_3.json");
        NetworkData network = io->read_network_data(network_file);

        // Verify structure for generic branch network
        std::cout << "Generic branch network structure:" << std::endl;
        std::cout << "  Buses: " << network.num_buses << std::endl;
        std::cout << "  Branches: " << network.num_branches << std::endl;
        std::cout << "  Appliances: " << network.num_appliances << std::endl;

        // Verify expected counts from network_3.json
        ASSERT_EQ(2, network.num_buses);       // nodes 1,2
        ASSERT_EQ(1, network.num_branches);    // generic_branch 3
        ASSERT_EQ(2, network.num_appliances);  // source 4, load 5

        // Verify bus voltage ratings
        std::cout << "\nGeneric branch bus details:" << std::endl;
        for (auto const& bus : network.buses) {
            std::cout << "  Bus " << bus.id << ": u_rated=" << bus.u_rated << "V" << std::endl;
        }

        // Check high voltage levels (380kV and 400kV)
        bool found_380kv = false, found_400kv = false;
        for (auto const& bus : network.buses) {
            if (std::abs(bus.u_rated - 380000.0) < 1e-6) found_380kv = true;
            if (std::abs(bus.u_rated - 400000.0) < 1e-6) found_400kv = true;
        }
        ASSERT_TRUE(found_380kv && found_400kv);

        // Verify generic branch data with direct PGM parameters
        std::cout << "\nGeneric branch details:" << std::endl;
        auto const& branch = network.branches[0];
        std::cout << "  Branch " << branch.id << ": " << branch.from_bus << "->" << branch.to_bus
                  << std::endl;
        std::cout << "    r1=" << branch.r1 << ", x1=" << branch.x1 << std::endl;
        std::cout << "    b1=" << branch.b1 << ", g1=" << branch.g1 << std::endl;
        std::cout << "    shift=" << branch.theta << ", k=" << branch.k << std::endl;
        std::cout << "    i_n=" << branch.i_n << ", type=" << static_cast<int>(branch.branch_type)
                  << std::endl;

        // Verify direct parameter mapping from network_3.json
        ASSERT_TRUE(branch.branch_type == BranchType::GENERIC);
        ASSERT_NEAR(0.1, branch.r1, 1e-6);     // Direct from r1
        ASSERT_NEAR(0.25, branch.x1, 1e-6);    // Direct from x1
        ASSERT_NEAR(0.0, branch.b1, 1e-6);     // Direct from b1
        ASSERT_NEAR(0.0, branch.g1, 1e-6);     // Direct from g1
        ASSERT_NEAR(0.0, branch.theta, 1e-6);  // Direct from shift
        ASSERT_NEAR(1.0526, branch.k, 1e-6);   // Direct from k
        ASSERT_TRUE(branch.i_n > 0.0);         // Calculated from sn

        // Verify appliances (source + load)
        std::cout << "\nAppliances in generic branch network:" << std::endl;
        bool found_source = false, found_load = false;
        for (auto const& appliance : network.appliances) {
            std::cout << "  " << (appliance.type == ApplianceType::SOURCE ? "Source" : "Load")
                      << " " << appliance.id << " at bus " << appliance.node << std::endl;
            if (appliance.type == ApplianceType::SOURCE) {
                found_source = true;
                ASSERT_NEAR(1.0, appliance.u_ref, 1e-6);
            } else if (appliance.type == ApplianceType::LOADGEN) {
                found_load = true;
                ASSERT_NEAR(100000.0, appliance.p_specified, 1e-6);  // 100 kW
                ASSERT_NEAR(2000.0, appliance.q_specified, 1e-6);    // 2 kVar
            }
        }
        ASSERT_TRUE(found_source && found_load);

        std::cout << "\n✓ Generic branch network test passed!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Generic branch test error: " << e.what() << std::endl;
        std::exit(1);
    }
}

void register_pgm_tests(TestRunner& runner) {
    runner.add_test("PGM JSON IO - Network 1 (Lines)", test_pgm_json_io);
    runner.add_test("PGM JSON IO - Network 2 (Transformer)", test_transformer_network);
    runner.add_test("PGM JSON IO - Network 3 (Generic Branch)", test_generic_branch_network);
}