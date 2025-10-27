#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "gap/io/io_interface.h"

// Simple JSON parser (basic implementation for PGM format)
// In production, would use nlohmann/json or similar library
namespace {

/**
 * @brief Simple JSON value class for PGM parsing
 */
class JsonValue {
  public:
    enum Type { STRING, NUMBER, OBJECT, ARRAY, BOOLEAN, NULL_VALUE };

  private:
    Type type_;
    std::string str_value_;
    double num_value_;
    std::unordered_map<std::string, JsonValue> obj_value_;
    std::vector<JsonValue> arr_value_;
    bool bool_value_;

  public:
    JsonValue() : type_(NULL_VALUE) {}
    JsonValue(const std::string& s) : type_(STRING), str_value_(s) {}
    JsonValue(double d) : type_(NUMBER), num_value_(d) {}
    JsonValue(bool b) : type_(BOOLEAN), bool_value_(b) {}

    Type getType() const { return type_; }

    const std::string& asString() const {
        if (type_ != STRING) throw std::runtime_error("Not a string");
        return str_value_;
    }

    double asNumber() const {
        if (type_ != NUMBER) throw std::runtime_error("Not a number");
        return num_value_;
    }

    bool asBool() const {
        if (type_ != BOOLEAN) throw std::runtime_error("Not a boolean");
        return bool_value_;
    }

    const std::unordered_map<std::string, JsonValue>& asObject() const {
        if (type_ != OBJECT) throw std::runtime_error("Not an object");
        return obj_value_;
    }

    const std::vector<JsonValue>& asArray() const {
        if (type_ != ARRAY) throw std::runtime_error("Not an array");
        return arr_value_;
    }

    void setObject(std::unordered_map<std::string, JsonValue> obj) {
        type_ = OBJECT;
        obj_value_ = std::move(obj);
    }

    void setArray(std::vector<JsonValue> arr) {
        type_ = ARRAY;
        arr_value_ = std::move(arr);
    }

    bool contains(const std::string& key) const {
        return type_ == OBJECT && obj_value_.find(key) != obj_value_.end();
    }

    const JsonValue& operator[](const std::string& key) const {
        if (type_ != OBJECT) throw std::runtime_error("Not an object");
        auto it = obj_value_.find(key);
        if (it == obj_value_.end()) throw std::runtime_error("Key not found: " + key);
        return it->second;
    }

    const JsonValue& operator[](size_t index) const {
        if (type_ != ARRAY) throw std::runtime_error("Not an array");
        if (index >= arr_value_.size()) throw std::runtime_error("Index out of range");
        return arr_value_[index];
    }

    size_t size() const {
        if (type_ == ARRAY) return arr_value_.size();
        if (type_ == OBJECT) return obj_value_.size();
        return 0;
    }
};

/**
 * @brief Simple JSON parser for PGM format
 */
JsonValue parseJson(const std::string& content);

}  // anonymous namespace

namespace gap::io {

/**
 * @brief Convert PGM node to GAP BusData
 */
BusData convertPgmNode(const JsonValue& pgm_node) {
    BusData bus;
    bus.id = static_cast<int>(pgm_node["id"].asNumber());
    bus.u_rated = pgm_node["u_rated"].asNumber();
    bus.bus_type = BusType::PQ;  // Default, will be updated based on connected appliances
    bus.energized = 0;           // Will be determined during network analysis
    bus.u = bus.u_rated;         // Initial guess
    bus.u_pu = 1.0;              // Initial guess
    bus.u_angle = 0.0;
    bus.active_power = 0.0;
    bus.reactive_power = 0.0;
    return bus;
}

/**
 * @brief Convert PGM line to GAP BranchData
 */
BranchData convertPgmLine(const JsonValue& pgm_line) {
    BranchData branch;
    branch.id = static_cast<int>(pgm_line["id"].asNumber());
    branch.from_bus = static_cast<int>(pgm_line["from_node"].asNumber());
    branch.to_bus = static_cast<int>(pgm_line["to_node"].asNumber());
    branch.from_status = static_cast<int8_t>(pgm_line["from_status"].asNumber());
    branch.to_status = static_cast<int8_t>(pgm_line["to_status"].asNumber());
    branch.status = (branch.from_status && branch.to_status) ? 1 : 0;

    // Electrical parameters
    branch.r1 = pgm_line["r1"].asNumber();
    branch.x1 = pgm_line["x1"].asNumber();

    // Convert capacitive parameters to admittance using our helper function
    if (pgm_line.contains("c1") && pgm_line.contains("tan1")) {
        double c1 = pgm_line["c1"].asNumber();
        double tan1 = pgm_line["tan1"].asNumber();
        branch.set_from_pgm_capacitive_params(c1, tan1, 50.0);  // 50Hz European grid
    }

    // Transformer parameters (defaults for lines)
    branch.k = 1.0;      // No tap ratio for lines
    branch.theta = 0.0;  // No phase shift for lines
    branch.sn = 0.0;     // No rating specified in PGM line

    // Rated current
    if (pgm_line.contains("i_n")) {
        branch.i_n = pgm_line["i_n"].asNumber();
    } else {
        branch.i_n = 1000.0;  // Default 1kA
    }

    return branch;
}

/**
 * @brief Convert PGM source to GAP ApplianceData
 */
ApplianceData convertPgmSource(const JsonValue& pgm_source) {
    ApplianceData appliance;
    appliance.id = static_cast<int>(pgm_source["id"].asNumber());
    appliance.node = static_cast<int>(pgm_source["node"].asNumber());
    appliance.status = static_cast<int8_t>(pgm_source["status"].asNumber());
    appliance.type = ApplianceType::SOURCE;

    // Source-specific parameters
    if (pgm_source.contains("u_ref")) {
        appliance.u_ref = pgm_source["u_ref"].asNumber();
    }
    if (pgm_source.contains("u_ref_angle")) {
        appliance.u_ref_angle = pgm_source["u_ref_angle"].asNumber();
    }
    if (pgm_source.contains("sk")) {
        appliance.sk = pgm_source["sk"].asNumber();
    }
    if (pgm_source.contains("rx_ratio")) {
        appliance.rx_ratio = pgm_source["rx_ratio"].asNumber();
    }

    return appliance;
}

/**
 * @brief Convert PGM load to GAP ApplianceData
 */
ApplianceData convertPgmLoad(const JsonValue& pgm_load) {
    ApplianceData appliance;
    appliance.id = static_cast<int>(pgm_load["id"].asNumber());
    appliance.node = static_cast<int>(pgm_load["node"].asNumber());
    appliance.status = static_cast<int8_t>(pgm_load["status"].asNumber());
    appliance.type = ApplianceType::LOADGEN;

    // Load/generator parameters
    if (pgm_load.contains("p_specified")) {
        appliance.p_specified = pgm_load["p_specified"].asNumber();
    }
    if (pgm_load.contains("q_specified")) {
        appliance.q_specified = pgm_load["q_specified"].asNumber();
    }

    return appliance;
}

/**
 * @brief Convert PGM transformer to GAP BranchData
 */
BranchData convertPgmTransformer(const JsonValue& pgm_transformer) {
    BranchData branch;
    branch.id = static_cast<int>(pgm_transformer["id"].asNumber());
    branch.from_bus = static_cast<int>(pgm_transformer["from_node"].asNumber());
    branch.to_bus = static_cast<int>(pgm_transformer["to_node"].asNumber());
    branch.from_status = static_cast<int8_t>(pgm_transformer["from_status"].asNumber());
    branch.to_status = static_cast<int8_t>(pgm_transformer["to_status"].asNumber());
    branch.status = (branch.from_status && branch.to_status) ? 1 : 0;
    branch.branch_type = BranchType::TRAFO;

    // Extract voltages and rated power
    double u1 = pgm_transformer["u1"].asNumber();  // Primary voltage (V)
    double u2 = pgm_transformer["u2"].asNumber();  // Secondary voltage (V)
    double sn = pgm_transformer["sn"].asNumber();  // Rated power (VA)
    double uk = pgm_transformer["uk"].asNumber();  // Short-circuit voltage (p.u.)

    // Calculate transformer parameters
    // uk is the short-circuit voltage in p.u. on the transformer rated power
    // For impedance calculation: Z_base = U²/S
    double z_base_primary = (u1 * u1) / sn;  // Base impedance on primary side (Ω)

    // Convert uk (p.u.) to actual impedance
    double z_total = uk * z_base_primary;  // Total impedance (Ω)

    // Assume typical transformer X/R ratio of 10 (unless specified)
    double xr_ratio = 10.0;
    if (pgm_transformer.contains("rx_ratio")) {
        xr_ratio = 1.0 / pgm_transformer["rx_ratio"].asNumber();  // rx_ratio is R/X, we need X/R
    }

    // Calculate R and X from total impedance
    // Z² = R² + X², X = xr_ratio * R
    // Z² = R² + (xr_ratio * R)² = R² * (1 + xr_ratio²)
    double r_factor = 1.0 / std::sqrt(1.0 + xr_ratio * xr_ratio);
    branch.r1 = z_total * r_factor;
    branch.x1 = z_total * xr_ratio * r_factor;

    // Transformer tap ratio
    branch.k = u1 / u2;  // Turns ratio (primary/secondary)

    // Handle tap changer if present
    if (pgm_transformer.contains("tap_pos") && pgm_transformer.contains("tap_size")) {
        double tap_pos = pgm_transformer["tap_pos"].asNumber();
        double tap_size = pgm_transformer["tap_size"].asNumber();  // Tap size in per mille
        // Modify tap ratio based on tap position (tap_size is typically in ‰)
        double tap_factor = 1.0 + (tap_pos * tap_size / 100000.0);
        branch.k *= tap_factor;
    }

    // Phase shift from vector group (clock position)
    if (pgm_transformer.contains("clock")) {
        double clock = pgm_transformer["clock"].asNumber();
        branch.theta = clock * M_PI / 6.0;  // Clock * 30° converted to radians
    } else {
        branch.theta = 0.0;
    }

    // Store rated power
    branch.sn = sn;

    // Calculate rated current (approximate)
    branch.i_n =
        sn / (std::sqrt(3.0) * std::min(u1, u2));  // Using lower voltage for current rating

    // Shunt parameters (simplified - could be enhanced with no-load data)
    branch.g1 = 0.0;  // Ignore magnetizing conductance for now
    branch.b1 = 0.0;  // Ignore magnetizing susceptance for now

    return branch;
}

/**
 * @brief Convert PGM generic_branch to GAP BranchData
 */
BranchData convertPgmGenericBranch(const JsonValue& pgm_generic) {
    BranchData branch;
    branch.id = static_cast<int>(pgm_generic["id"].asNumber());
    branch.from_bus = static_cast<int>(pgm_generic["from_node"].asNumber());
    branch.to_bus = static_cast<int>(pgm_generic["to_node"].asNumber());
    branch.from_status = static_cast<int8_t>(pgm_generic["from_status"].asNumber());
    branch.to_status = static_cast<int8_t>(pgm_generic["to_status"].asNumber());
    branch.status = (branch.from_status && branch.to_status) ? 1 : 0;
    branch.branch_type = BranchType::GENERIC;

    // Direct parameter mapping (generic branch has explicit r1, x1, g1, b1)
    branch.r1 = pgm_generic["r1"].asNumber();
    branch.x1 = pgm_generic["x1"].asNumber();

    if (pgm_generic.contains("g1")) {
        branch.g1 = pgm_generic["g1"].asNumber();
    }
    if (pgm_generic.contains("b1")) {
        branch.b1 = pgm_generic["b1"].asNumber();
    }

    // Tap ratio
    if (pgm_generic.contains("k")) {
        branch.k = pgm_generic["k"].asNumber();
    }

    // Phase shift (PGM uses "shift" in radians)
    if (pgm_generic.contains("shift")) {
        branch.theta = pgm_generic["shift"].asNumber();
    }

    // Rated power
    if (pgm_generic.contains("sn")) {
        branch.sn = pgm_generic["sn"].asNumber();
    }

    // Rated current (if not specified, calculate from sn and approximate voltage)
    if (pgm_generic.contains("i_n")) {
        branch.i_n = pgm_generic["i_n"].asNumber();
    } else if (branch.sn > 0.0) {
        // Approximate rated current (assuming typical medium voltage)
        branch.i_n = branch.sn / (std::sqrt(3.0) * 20000.0);  // Assume 20kV for approximation
    }

    return branch;
}

/**
 * @brief Determine bus types based on connected appliances
 */
void inferBusTypes(NetworkData& network) {
    // Create mapping of bus to connected appliances
    std::unordered_map<int, std::vector<ApplianceData*>> bus_appliances;

    for (auto& appliance : network.appliances) {
        bus_appliances[appliance.node].push_back(&appliance);
    }

    // Determine bus type for each bus
    for (auto& bus : network.buses) {
        auto it = bus_appliances.find(bus.id);
        if (it == bus_appliances.end()) {
            // No appliances connected - keep as PQ bus
            continue;
        }

        bool has_source = false;
        bool has_pv_gen = false;

        for (const auto* appliance : it->second) {
            if (appliance->type == ApplianceType::SOURCE) {
                has_source = true;
            } else if (appliance->type == ApplianceType::LOADGEN && appliance->p_specified < 0) {
                // Negative P indicates generation
                has_pv_gen = true;
            }
        }

        if (has_source) {
            bus.bus_type = BusType::SLACK;  // Source bus becomes slack
        } else if (has_pv_gen) {
            bus.bus_type = BusType::PV;  // Generator bus becomes PV
        } else {
            bus.bus_type = BusType::PQ;  // Load bus remains PQ
        }
    }
}

class JsonIOModule : public IIOModule {
  public:
    NetworkData read_network_data(const std::string& filename) override {
        std::cout << "JsonIOModule: Reading PGM network data from " << filename << std::endl;

        // Read file content
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        // Parse JSON
        JsonValue root;
        try {
            root = parseJson(content);
        } catch (const std::exception& e) {
            throw std::runtime_error("JSON parsing failed: " + std::string(e.what()));
        }

        // Extract PGM metadata
        NetworkData network;
        if (root.contains("version")) {
            network.version = root["version"].asString();
        }
        if (root.contains("type")) {
            network.type = root["type"].asString();
        }
        if (root.contains("is_batch")) {
            network.is_batch = root["is_batch"].asBool();
        }

        // Extract data section
        const JsonValue& data_section = root["data"];

        // Convert PGM nodes to GAP buses
        if (data_section.contains("node")) {
            const JsonValue& nodes = data_section["node"];
            for (size_t i = 0; i < nodes.size(); ++i) {
                BusData bus = convertPgmNode(nodes[i]);
                network.buses.push_back(bus);
            }
        }

        // Convert PGM lines to GAP branches
        if (data_section.contains("line")) {
            const JsonValue& lines = data_section["line"];
            for (size_t i = 0; i < lines.size(); ++i) {
                BranchData branch = convertPgmLine(lines[i]);
                network.branches.push_back(branch);
            }
        }

        // Convert PGM transformers to GAP branches
        if (data_section.contains("transformer")) {
            const JsonValue& transformers = data_section["transformer"];
            for (size_t i = 0; i < transformers.size(); ++i) {
                BranchData branch = convertPgmTransformer(transformers[i]);
                network.branches.push_back(branch);
            }
        }

        // Convert PGM generic branches to GAP branches
        if (data_section.contains("generic_branch")) {
            const JsonValue& generic_branches = data_section["generic_branch"];
            for (size_t i = 0; i < generic_branches.size(); ++i) {
                BranchData branch = convertPgmGenericBranch(generic_branches[i]);
                network.branches.push_back(branch);
            }
        }

        // Convert PGM sources to GAP appliances
        if (data_section.contains("source")) {
            const JsonValue& sources = data_section["source"];
            for (size_t i = 0; i < sources.size(); ++i) {
                ApplianceData appliance = convertPgmSource(sources[i]);
                network.appliances.push_back(appliance);
            }
        }

        // Convert PGM loads to GAP appliances
        if (data_section.contains("sym_load")) {
            const JsonValue& loads = data_section["sym_load"];
            for (size_t i = 0; i < loads.size(); ++i) {
                ApplianceData appliance = convertPgmLoad(loads[i]);
                network.appliances.push_back(appliance);
            }
        }

        // Convert PGM generators (if present)
        if (data_section.contains("sym_gen")) {
            const JsonValue& generators = data_section["sym_gen"];
            for (size_t i = 0; i < generators.size(); ++i) {
                ApplianceData appliance = convertPgmLoad(generators[i]);  // Same structure as load
                network.appliances.push_back(appliance);
            }
        }

        // Infer bus types from connected appliances
        inferBusTypes(network);

        // Set counts
        network.num_buses = static_cast<int>(network.buses.size());
        network.num_branches = static_cast<int>(network.branches.size());
        network.num_appliances = static_cast<int>(network.appliances.size());

        // Validate PGM compliance
        if (!network.validate_pgm_compliance()) {
            std::cerr << "Warning: Network data does not meet PGM compliance standards"
                      << std::endl;
        }

        std::cout << "  Loaded " << network.num_buses << " buses, " << network.num_branches
                  << " branches, " << network.num_appliances << " appliances" << std::endl;

        return network;
    }

    void write_results(const std::string& filename, const ComplexVector& bus_voltages,
                       bool converged, int iterations) override {
        // TODO: Implement JSON output for results
        std::cout << "JsonIOModule: Writing results to " << filename << std::endl;
        std::cout << "  Converged: " << (converged ? "Yes" : "No") << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Bus voltages count: " << bus_voltages.size() << std::endl;

        // Placeholder implementation
        // In real implementation, format and write JSON output
    }

    bool validate_input_format(const std::string& filename) override {
        // TODO: Implement JSON format validation
        std::cout << "JsonIOModule: Validating format of " << filename << std::endl;

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return false;
        }

        // Placeholder implementation
        // In real implementation, validate JSON schema
        return true;
    }
};

}  // namespace gap::io

// Simple JSON parser implementation
namespace {

void skipWhitespace(const std::string& str, size_t& pos) {
    while (pos < str.length() && std::isspace(str[pos])) {
        ++pos;
    }
}

std::string parseString(const std::string& str, size_t& pos) {
    if (str[pos] != '"') {
        throw std::runtime_error("Expected '\"' at position " + std::to_string(pos));
    }
    ++pos;  // Skip opening quote

    std::string result;
    while (pos < str.length() && str[pos] != '"') {
        if (str[pos] == '\\') {
            ++pos;  // Skip escape character
            if (pos >= str.length()) {
                throw std::runtime_error("Unexpected end of string");
            }
            // Handle basic escape sequences
            switch (str[pos]) {
                case '"':
                    result += '"';
                    break;
                case '\\':
                    result += '\\';
                    break;
                case '/':
                    result += '/';
                    break;
                case 'n':
                    result += '\n';
                    break;
                case 'r':
                    result += '\r';
                    break;
                case 't':
                    result += '\t';
                    break;
                default:
                    result += str[pos];
                    break;
            }
        } else {
            result += str[pos];
        }
        ++pos;
    }

    if (pos >= str.length()) {
        throw std::runtime_error("Unterminated string");
    }
    ++pos;  // Skip closing quote

    return result;
}

double parseNumber(const std::string& str, size_t& pos) {
    size_t start = pos;

    // Handle optional negative sign
    if (pos < str.length() && str[pos] == '-') {
        ++pos;
    }

    // Parse integer part
    if (pos >= str.length() || !std::isdigit(str[pos])) {
        throw std::runtime_error("Invalid number format");
    }

    while (pos < str.length() && std::isdigit(str[pos])) {
        ++pos;
    }

    // Handle decimal part
    if (pos < str.length() && str[pos] == '.') {
        ++pos;
        while (pos < str.length() && std::isdigit(str[pos])) {
            ++pos;
        }
    }

    // Handle exponent (scientific notation like 0.72e-7)
    if (pos < str.length() && (str[pos] == 'e' || str[pos] == 'E')) {
        ++pos;
        if (pos < str.length() && (str[pos] == '+' || str[pos] == '-')) {
            ++pos;
        }
        while (pos < str.length() && std::isdigit(str[pos])) {
            ++pos;
        }
    }

    std::string numStr = str.substr(start, pos - start);
    return std::stod(numStr);
}

JsonValue parseValue(const std::string& str, size_t& pos);

JsonValue parseArray(const std::string& str, size_t& pos) {
    JsonValue result;
    std::vector<JsonValue> arr;

    if (str[pos] != '[') {
        throw std::runtime_error("Expected '[' at position " + std::to_string(pos));
    }
    ++pos;  // Skip '['

    skipWhitespace(str, pos);

    if (pos < str.length() && str[pos] == ']') {
        ++pos;  // Empty array
        result.setArray(std::move(arr));
        return result;
    }

    while (pos < str.length()) {
        skipWhitespace(str, pos);
        arr.push_back(parseValue(str, pos));
        skipWhitespace(str, pos);

        if (pos >= str.length()) {
            throw std::runtime_error("Unterminated array");
        }

        if (str[pos] == ']') {
            ++pos;
            break;
        } else if (str[pos] == ',') {
            ++pos;
        } else {
            throw std::runtime_error("Expected ',' or ']' in array");
        }
    }

    result.setArray(std::move(arr));
    return result;
}

JsonValue parseObject(const std::string& str, size_t& pos) {
    JsonValue result;
    std::unordered_map<std::string, JsonValue> obj;

    if (str[pos] != '{') {
        throw std::runtime_error("Expected '{' at position " + std::to_string(pos));
    }
    ++pos;  // Skip '{'

    skipWhitespace(str, pos);

    if (pos < str.length() && str[pos] == '}') {
        ++pos;  // Empty object
        result.setObject(std::move(obj));
        return result;
    }

    while (pos < str.length()) {
        skipWhitespace(str, pos);

        // Parse key
        std::string key = parseString(str, pos);
        skipWhitespace(str, pos);

        if (pos >= str.length() || str[pos] != ':') {
            throw std::runtime_error("Expected ':' after key");
        }
        ++pos;  // Skip ':'

        skipWhitespace(str, pos);

        // Parse value
        JsonValue value = parseValue(str, pos);
        obj[key] = std::move(value);

        skipWhitespace(str, pos);

        if (pos >= str.length()) {
            throw std::runtime_error("Unterminated object");
        }

        if (str[pos] == '}') {
            ++pos;
            break;
        } else if (str[pos] == ',') {
            ++pos;
        } else {
            throw std::runtime_error("Expected ',' or '}' in object");
        }
    }

    result.setObject(std::move(obj));
    return result;
}

JsonValue parseValue(const std::string& str, size_t& pos) {
    skipWhitespace(str, pos);

    if (pos >= str.length()) {
        throw std::runtime_error("Unexpected end of input");
    }

    char c = str[pos];

    if (c == '"') {
        return JsonValue(parseString(str, pos));
    } else if (c == '{') {
        return parseObject(str, pos);
    } else if (c == '[') {
        return parseArray(str, pos);
    } else if (std::isdigit(c) || c == '-') {
        return JsonValue(parseNumber(str, pos));
    } else if (str.substr(pos, 4) == "true") {
        pos += 4;
        return JsonValue(true);
    } else if (str.substr(pos, 5) == "false") {
        pos += 5;
        return JsonValue(false);
    } else if (str.substr(pos, 4) == "null") {
        pos += 4;
        return JsonValue();  // NULL_VALUE
    } else {
        throw std::runtime_error("Unexpected character '" + std::string(1, c) + "' at position " +
                                 std::to_string(pos));
    }
}

JsonValue parseJson(const std::string& content) {
    size_t pos = 0;
    JsonValue result = parseValue(content, pos);
    skipWhitespace(content, pos);

    if (pos < content.length()) {
        throw std::runtime_error("Extra characters after JSON");
    }

    return result;
}

}  // anonymous namespace
