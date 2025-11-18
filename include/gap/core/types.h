#pragma once

#include <cmath>  // for M_PI in PGM conversion functions
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace gap {

/**
 * @brief GAP global precision
 *
 */
using Float = double;

/**
 * @brief Complex number type alias for power flow calculations
 */
using Complex = std::complex<Float>;

/**
 * @brief Vector of complex numbers
 */
using ComplexVector = std::vector<Complex>;

/**
 * @brief Bus type enumeration for power system analysis
 */
enum class BusType {
    PQ = 0,    // Load bus (P and Q specified) - Most common in PGM
    PV = 1,    // Generator bus (P and V specified) - Not present in PGM
    SLACK = 2  // Slack/reference bus (V and angle specified)
};

/**
 * @brief Appliance type
 *
 */
enum class ApplianceType {
    SOURCE = 0,   // Source appliance
    LOADGEN = 1,  // Load appliance
    SHUNT = 2     // Shunt appliance (capacitor/reactor)
};

/**
 * @brief Branch type enumeration for power system analysis
 *
 */
enum class BranchType {
    LINE = 0,    // Line
    TRAFO = 1,   // Transformer
    GENERIC = 2  // Generic branch
};

/**
 * @brief Sparse matrix structure for admittance matrices
 */
struct SparseMatrix {
    std::vector<int> row_ptr;     // Row pointers (CSR format)
    std::vector<int> col_idx;     // Column indices
    std::vector<Complex> values;  // Complex values
    int num_rows;                 // Number of rows
    int num_cols;                 // Number of columns
    int nnz;                      // Number of non-zero elements
};

#ifdef __CUDACC__
#include <cuComplex.h>
#endif

/**
 * @brief GPU-resident sparse matrix structure
 *
 * Maintains both device pointers for computation and optional host shadow
 * for debugging/verification. The host shadow can be synchronized on demand.
 */
struct GPUSparseMatrix {
    // Device memory pointers
    int* d_row_ptr{nullptr};
    int* d_col_idx{nullptr};
#ifdef __CUDACC__
    cuDoubleComplex* d_values{nullptr};
#else
    void* d_values{nullptr};  // Opaque pointer when not compiling with nvcc
#endif

    // Matrix dimensions
    int num_rows{0};
    int num_cols{0};
    int nnz{0};

    // Host shadow copy for debugging (optional)
    bool has_host_shadow{false};
    std::vector<int> h_row_ptr;
    std::vector<int> h_col_idx;
    std::vector<Complex> h_values;

    // Memory ownership flag
    bool owns_device_memory{true};

    /**
     * @brief Synchronize host shadow from device
     * Copies current device data to host vectors for debugging
     */
    void sync_from_device();

    /**
     * @brief Synchronize device from host shadow
     * Copies host vectors to device memory
     */
    void sync_to_device();

    /**
     * @brief Allocate device memory
     */
    void allocate_device(int rows, int cols, int nonzeros);

    /**
     * @brief Free device memory
     */
    void free_device();

    /**
     * @brief Get host copy for verification (creates shadow if needed)
     */
    SparseMatrix get_host_copy();
};

/**
 * @brief GPU-resident power flow data
 *
 * Maintains all vectors needed for Newton-Raphson iterations on GPU,
 * with optional synchronization to CPU for debugging.
 */
struct GPUPowerFlowData {
#ifdef __CUDACC__
    // Device vectors
    cuDoubleComplex* d_voltages{nullptr};          // Bus voltage phasors
    cuDoubleComplex* d_power_injections{nullptr};  // S_specified = P + jQ
    double* d_mismatches{nullptr};                 // Power mismatches (P and Q)
    cuDoubleComplex* d_rhs{nullptr};               // RHS for linear system
    cuDoubleComplex* d_solution{nullptr};          // Solution delta V
    int* d_bus_types{nullptr};                     // Bus type array
#else
    void* d_voltages{nullptr};
    void* d_power_injections{nullptr};
    void* d_mismatches{nullptr};
    void* d_rhs{nullptr};
    void* d_solution{nullptr};
    void* d_bus_types{nullptr};
#endif

    // Dimensions
    int num_buses{0};
    int num_pq_buses{0};
    int num_pv_buses{0};
    int num_unknowns{0};  // 2*num_pq + num_pv

    // Host shadow for debugging
    bool has_host_shadow{false};
    std::vector<Complex> h_voltages;
    std::vector<Complex> h_power_injections;
    std::vector<Float> h_mismatches;
    std::vector<Complex> h_rhs;
    std::vector<Complex> h_solution;
    std::vector<int> h_bus_types;

    // Memory ownership
    bool owns_device_memory{true};

    /**
     * @brief Allocate device memory for all vectors
     */
    void allocate_device(int buses, int unknowns);

    /**
     * @brief Free all device memory
     */
    void free_device();

    /**
     * @brief Sync specific vector from device for debugging
     */
    void sync_voltages_from_device();
    void sync_mismatches_from_device();
    void sync_solution_from_device();

    /**
     * @brief Get snapshot of all state for verification
     */
    struct StateSnapshot {
        std::vector<Complex> voltages;
        std::vector<Float> mismatches;
        Float max_mismatch;
        int iteration;
    };
    StateSnapshot get_state_snapshot();
};

/**
 * @brief Power system bus data - PGM Node equivalent
 */
struct BusData {
    // === Input Fields (Required) ===
    int id;         // Bus ID - must be unique
    Float u_rated;  // Nominal voltage (V) - must be > 0

    // === Input Fields (With Defaults) ===
    BusType bus_type{BusType::PQ};  // Bus type - default to PQ bus

    // === Output Fields (Power Flow Results) ===
    int8_t energized{0};        // Bus connected to a source (output)
    Float u{0.0};               // Voltage magnitude (V)
    Float u_pu{0.0};            // Voltage magnitude (p.u.)
    Float u_angle{0.0};         // Voltage angle (radians)
    Float active_power{0.0};    // Active power injection (W) - generator direction
    Float reactive_power{0.0};  // Reactive power injection (VAr) - generator direction
};

/**
 * @brief Power system generic branch data
 */
struct BranchData {
    int id;                                    // branch ID
    int from_bus;                              // From bus ID
    int8_t from_status;                        // From bus status (1=connected, 0=disconnected)
    int to_bus;                                // To bus ID
    int8_t to_status;                          // To bus status (1=connected, 0=disconnected)
    int8_t status;                             // Branch status (1=in-service, 0=out-of-service)
    BranchType branch_type{BranchType::LINE};  // Branch type (0=line, 1=transformer)

    Float p_from{0.0};        // Active power flowing into the branch at from-side
    Float p_to{0.0};          // Active power flowing into the branch at to-side
    Float q_from{0.0};        // Reactive power flowing into the branch at from-side
    Float q_to{0.0};          // Reactive power flowing into the branch at to-side
    Float s_from{0.0};        // Apparent power flowing into the branch at from-side
    Float s_to{0.0};          // Apparent power flowing into the branch at to-side
    Float i_from{0.0};        // Magnitude of current at from-side
    Float i_from_angle{0.0};  // Angle of current at from-side
    Float i_to{0.0};          // Magnitude of current at to-side
    Float i_to_angle{0.0};    // Angle of current at to-side
    Float loading{0.0};       // Relative loading of the branch

    // === Electrical Parameters ===
    Float r1{1e-6};    // positive-sequence serial resistance (ohm Ω) - small non-zero default
    Float x1{1e-6};    // positive-sequence serial reactance (ohm Ω) - small non-zero default
    Float g1{0.0};     // positive-sequence conductance (siemens)
    Float b1{0.0};     // positive-sequence susceptance (siemens)
    Float k{1.0};      // off-nominal ratio (tap) - PGM default 1.0
    Float theta{0.0};  // angle shift (radians) - PGM default 0.0
    Float sn{0.0};     // rated power (VA) - PGM default 0.0

    // === PGM Line/Transformer Parameters ===
    Float i_n{1000.0};  // Rated current (A) - for loading calculation and thermal limits

    // === PGM Conversion Helpers ===
    // Note: c1 (capacitance) and tan1 (loss factor) from PGM are converted to g1,b1 during JSON
    // parsing Formula: b1 = 2*π*f*c1, g1 = b1*tan1 (where f=50Hz for European grid)

    /**
     * @brief Convert PGM line capacitive parameters to admittance
     * @param c1 Line capacitance (F)
     * @param tan1 Loss factor (-)
     * @param frequency Grid frequency (Hz), default 50Hz
     */
    void set_from_pgm_capacitive_params(Float c1, Float tan1, Float frequency = 50.0) {
        if (c1 > 0) {
            Float omega = 2.0 * M_PI * frequency;
            b1 = omega * c1;  // Susceptance from capacitance
            g1 = b1 * tan1;   // Conductance from loss factor
        }
    }
};

/**
 * @brief Generic power system appliance data (PGM-compliant design)
 *
 * Represents any single-port component connected to a bus:
 * - SOURCE: External network equivalent (Thevenin)
 * - LOADGEN: Active/reactive power injection
 * - SHUNT: Fixed admittance element
 */
struct ApplianceData {
    int id;              // Appliance ID
    int node;            // Connected bus/node ID
    int8_t status;       // Connection status (1=connected, 0=disconnected)
    ApplianceType type;  // Appliance type (SOURCE, LOADGEN, SHUNT)

    // === Common Power Flow Output ===
    Float p{0.0};  // Active power (W) - generator reference direction
    Float q{0.0};  // Reactive power (VAr) - generator reference direction
    Float i{0.0};  // Current magnitude (A)
    Float s{0.0};  // Apparent power (VA)

    // === Type-Specific Parameters ===
    // SOURCE parameters
    Float u_ref{1.0};        // Reference voltage (p.u.) - for sources
    Float u_ref_angle{0.0};  // Reference voltage angle (rad) - for sources
    Float sk{1e10};          // Short circuit power (VA) - for sources
    Float rx_ratio{0.1};     // R/X ratio - for sources

    // LOADGEN parameters
    Float p_specified{0.0};  // Specified active power (W) - for loads/generators
    Float q_specified{0.0};  // Specified reactive power (VAr) - for loads/generators

    // SHUNT parameters
    Float g1{0.0};  // Positive-sequence conductance (S) - for shunts
    Float b1{0.0};  // Positive-sequence susceptance (S) - for shunts
};

/**
 * @brief Power system network data container
 */
/**
 * @brief Network data structure following Power Grid Model paradigm
 * Contains all power system components with PGM-compliant metadata
 */
struct NetworkData {
    // === Core Components ===
    std::vector<BusData> buses;
    std::vector<BranchData> branches;
    std::vector<ApplianceData> appliances;
    int num_buses;       // Number of buses
    int num_branches;    // Number of branches
    int num_appliances;  // Number of appliances (sources, loads, shunts)

    // === PGM JSON Metadata ===
    std::string version{"1.0"};  // PGM format version for JSON compatibility
    std::string type{"input"};   // Dataset type identifier
    bool is_batch{false};        // Batch calculation flag for PGM

    /**
     * @brief Validate PGM compliance and component consistency
     * @return true if all components have valid PGM parameters
     */
    bool validate_pgm_compliance() const {
        // Check all buses have positive rated voltage
        for (const auto& bus : buses) {
            if (bus.u_rated <= 0.0) return false;
        }

        // Check all branches have valid electrical parameters
        for (const auto& branch : branches) {
            if (branch.r1 < 0.0 || branch.x1 == 0.0) return false;  // x1 must be non-zero
            if (branch.i_n <= 0.0) return false;  // Rated current must be positive
        }

        // All validation checks passed
        return true;
    }
};

/**
 * @brief Backend execution type enumeration
 */
enum class BackendType { CPU, GPU_CUDA /*, GPU_HIP*/ };

/**
 * @brief Convert BusType enum to string
 */
inline std::string bus_type_to_string(BusType type) {
    switch (type) {
        case BusType::PQ:
            return "PQ";
        case BusType::PV:
            return "PV";
        case BusType::SLACK:
            return "SLACK";
        default:
            return "Unknown";
    }
}

/**
 * @brief Convert ApplianceType enum to string
 */
inline std::string appliance_type_to_string(ApplianceType type) {
    switch (type) {
        case ApplianceType::SOURCE:
            return "SOURCE";
        case ApplianceType::LOADGEN:
            return "LOADGEN";
        case ApplianceType::SHUNT:
            return "SHUNT";
        default:
            return "Unknown";
    }
}

/**
 * @brief Convert BackendType enum to string
 */
inline std::string backend_type_to_string(BackendType type) {
    switch (type) {
        case BackendType::CPU:
            return "CPU";
        case BackendType::GPU_CUDA:
            return "GPU_CUDA";
        /*
        case BackendType::GPU_HIP:
            return "GPU_HIP";
        */
        default:
            return "Unknown";
    }
}

}  // namespace gap