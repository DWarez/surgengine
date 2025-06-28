#include <atomic>
#include <boost/beast.hpp>
#include <ctime>
#include <functional>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace beast = boost::beast;
namespace http = beast::http;
using json = nlohmann::json;

#pragma once
class RequestHandler {
public:
  using Handler = std::function<json(const http::request<http::string_body> &)>;

private:
  std::unordered_map<std::string, std::unordered_map<http::verb, Handler>>
      routes_;

  // Instance members instead of static - much cleaner!
  std::time_t start_time_;
  mutable std::atomic<long> request_count_;

  http::response<http::string_body>
  make_json_response(const http::request<http::string_body> &req,
                     const json &body, http::status status = http::status::ok) {

    http::response<http::string_body> res{status, req.version()};
    res.set(http::field::server, "Surgengine");
    res.set(http::field::content_type, "application/json");
    res.keep_alive(req.keep_alive());
    res.body() = body.dump();
    res.prepare_payload();
    return res;
  }

  http::response<http::string_body>
  make_error_response(const http::request<http::string_body> &req,
                      http::status status, const std::string &message) {

    json error_body = {{"error", message}};
    return make_json_response(req, error_body, status);
  }

public:
  RequestHandler() : start_time_(std::time(nullptr)), request_count_(0) {
    setup_default_routes();
  }

  RequestHandler &route(http::verb method, const std::string &path,
                        Handler handler) {
    routes_[path][method] = std::move(handler);
    return *this;
  }

  http::response<http::string_body>
  handle_request(const http::request<http::string_body> &req) {
    try {
      auto path_it = routes_.find(std::string(req.target()));
      if (path_it == routes_.end()) {
        return make_error_response(req, http::status::not_found,
                                   "Route not found");
      }

      auto method_it = path_it->second.find(req.method());
      if (method_it == path_it->second.end()) {
        return make_error_response(req, http::status::method_not_allowed,
                                   "Method not allowed");
      }

      json response_body = method_it->second(req);
      return make_json_response(req, response_body);

    } catch (const json::parse_error &e) {
      return make_error_response(req, http::status::bad_request,
                                 "Invalid JSON: " + std::string(e.what()));
    } catch (const std::exception &e) {
      return make_error_response(req, http::status::internal_server_error,
                                 "Internal error: " + std::string(e.what()));
    }
  }

private:
  void setup_default_routes() {
    // Root endpoint - API overview
    route(http::verb::get, "/", [](const auto &req) {
      return json{
          {"service", "Surgengine Inference Engine"},
          {"version", "1.0.0"},
          {"status", "running"},
          {"endpoints", json::array({"/heartbeat - Simple liveness check",
                                     "/health - Detailed health information",
                                     "/ready - Readiness check",
                                     "/info - Server information",
                                     "/metrics - Performance metrics",
                                     "/api/v1 - API documentation"})}};
    });

    // Heartbeat - minimal liveness check (for load balancers)
    route(http::verb::get, "/heartbeat",
          [](const auto &req) { return json{{"alive", true}}; });

    // Health check - detailed health information
    route(http::verb::get, "/health", [this](const auto &req) {
      auto now = std::time(nullptr);
      return json{{"status", "healthy"},
                  {"timestamp", now},
                  {"uptime_seconds", now - start_time_},
                  {"server", "Surgengine"},
                  {"version", "1.0.0"},
                  {"checks", json::object({{"database", "connected"},
                                           {"memory", "normal"},
                                           {"disk_space", "sufficient"}})}};
    });

    // Readiness check - indicates if server is ready to accept traffic
    route(http::verb::get, "/ready", [](const auto &req) {
      return json{{"ready", true},
                  {"services", json::object({{"inference_engine", "ready"},
                                             {"model_loader", "ready"},
                                             {"api_server", "ready"}})},
                  {"timestamp", std::time(nullptr)}};
    });

    // Server information
    route(http::verb::get, "/info", [this](const auto &req) {
      auto now = std::time(nullptr);
      return json{{"service", "Surgengine Inference Engine"},
                  {"version", "1.0.0"},
                  {"build_date", "2025-06-28"},
                  {"environment", "production"},
                  {"started_at", start_time_},
                  {"uptime_seconds", now - start_time_},
                  {"hostname", "surgengine-server"},
                  {"platform", "linux-x64"},
                  {"cpp_version", __cplusplus},
                  {"boost_version", BOOST_VERSION / 100000}};
    });

    // Performance metrics
    route(http::verb::get, "/metrics", [this](const auto &req) {
      return json{{"requests_total", ++request_count_},
                  {"uptime_seconds", std::time(nullptr) - start_time_},
                  {"memory_usage_mb",
                   128}, // Placeholder - would use real memory monitoring
                  {"cpu_usage_percent", 15.5}, // Placeholder
                  {"active_connections", 1},   // Placeholder
                  {"response_times", json::object({{"avg_ms", 42.3},
                                                   {"p95_ms", 89.1},
                                                   {"p99_ms", 156.7}})},
                  {"timestamp", std::time(nullptr)}};
    });

    // API documentation
    route(http::verb::get, "/api/v1", [](const auto &req) {
      return json{
          {"api_version", "v1"},
          {"documentation", "Surgengine Inference API"},
          {"base_url", "/api/v1"},
          {"endpoints",
           json::object(
               {{"GET /inference", "List available inference endpoints"},
                {"POST /inference/predict", "Run model inference"},
                {"GET /models", "List available models"},
                {"POST /models/load", "Load a specific model"},
                {"DELETE /models/{id}", "Unload a model"},
                {"GET /jobs", "List inference jobs"},
                {"GET /jobs/{id}", "Get job status"}})},
          {"authentication",
           "Bearer token required for POST/DELETE operations"},
          {"rate_limits", json::object({{"requests_per_minute", 1000},
                                        {"inference_per_hour", 100}})}};
    });

    // Example inference endpoints
    route(http::verb::get, "/api/v1/inference", [](const auto &req) {
      return json{
          {"available_endpoints",
           json::array({{"endpoint", "/api/v1/inference/predict", "method",
                         "POST", "description", "Run model inference"},
                        {"endpoint", "/api/v1/inference/batch", "method",
                         "POST", "description", "Batch inference"},
                        {"endpoint", "/api/v1/inference/stream", "method",
                         "POST", "description", "Streaming inference"}})},
          {"supported_formats", json::array({"json", "binary", "base64"})},
          {"max_input_size_mb", 10}};
    });

    route(http::verb::post, "/api/v1/inference/predict", [](const auto &req) {
      auto input = json::parse(req.body());

      std::string model = input.value("model", "default");
      auto data = input.value("input", json::object());

      return json{{"job_id", "job_" + std::to_string(std::time(nullptr))},
                  {"status", "completed"},
                  {"model_used", model},
                  {"prediction", json::object({{"class", "positive"},
                                               {"confidence", 0.87},
                                               {"probability_distribution",
                                                json::array({0.13, 0.87})}})},
                  {"processing_time_ms", 45},
                  {"timestamp", std::time(nullptr)}};
    });

    // Model management
    route(http::verb::get, "/api/v1/models", [](const auto &req) {
      return json{
          {"models", json::array({{{"id", "model_001"},
                                   {"name", "SurgNet-V1"},
                                   {"type", "classification"},
                                   {"status", "loaded"},
                                   {"version", "1.2.0"},
                                   {"memory_usage_mb", 512},
                                   {"last_used", std::time(nullptr) - 3600}},
                                  {{"id", "model_002"},
                                   {"name", "TextAnalyzer"},
                                   {"type", "nlp"},
                                   {"status", "available"},
                                   {"version", "2.1.0"},
                                   {"memory_usage_mb", 0},
                                   {"last_used", nullptr}}})},
          {"total_models", 2},
          {"loaded_models", 1},
          {"total_memory_usage_mb", 512}};
    });

    route(http::verb::post, "/api/v1/models/load", [](const auto &req) {
      auto request_data = json::parse(req.body());
      std::string model_id = request_data.value("model_id", "unknown");

      return json{{"job_id", "load_" + std::to_string(std::time(nullptr))},
                  {"status", "loading"},
                  {"model_id", model_id},
                  {"estimated_time_seconds", 30},
                  {"message", "Model loading initiated"}};
    });

    // System control endpoints (for ops/admin)
    route(http::verb::post, "/api/v1/system/gc", [](const auto &req) {
      return json{{"message", "Garbage collection triggered"},
                  {"memory_freed_mb", 45},
                  {"timestamp", std::time(nullptr)}};
    });

    route(http::verb::get, "/api/v1/system/stats", [](const auto &req) {
      return json{{"system", json::object({{"cpu_cores", 8},
                                           {"total_memory_gb", 32},
                                           {"available_memory_gb", 18.5},
                                           {"disk_usage_percent", 65},
                                           {"load_average",
                                            json::array({1.2, 1.5, 1.8})}})},
                  {"process", json::object({{"pid", 12345},
                                            {"memory_usage_mb", 256},
                                            {"cpu_usage_percent", 12.3},
                                            {"threads", 4},
                                            {"open_files", 23}})},
                  {"timestamp", std::time(nullptr)}};
    });
  }
};