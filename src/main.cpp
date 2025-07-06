#include <boost/asio/basic_socket_acceptor.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/socket_base.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/core/error.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/message_fwd.hpp>
#include <boost/beast/http/string_body_fwd.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <core/device.cuh>
#include <core/nn/module.cuh>
#include <core/nn/parameter.cuh>
#include <core/tensor.cuh>
#include <memory>
#include <nlohmann/json.hpp>
#include <server/server.hpp>

using namespace surgengine;
using namespace surgengine::nn;

int main() {
  // Module<float> module("test_module", Device::cpu());
  // module.register_parameter("culo",
  // std::make_shared<Parameter<float>>("culo")); auto submodule =
  // std::make_shared<Module<float>>("sub");
  // submodule->register_parameter("subculo",
  //                               std::make_shared<Parameter<float>>("subculo"));
  // module.register_module("sub", submodule);
  // auto all = module.named_parameters();
  return 0;
  // std::cout << "Starting Surgengine server..." << std::endl;

  // try {
  //   // Create server on port 8080
  //   Server server(8080);

  //   if (!server.is_initialized()) {
  //     std::cerr << "Failed to initialize server" << std::endl;
  //     return 1;
  //   }

  //   // Optionally add custom business logic routes
  //   server.routes()
  //       .route(http::verb::post, "/api/v1/inference/batch",
  //              [](const auto &req) {
  //                auto input = json::parse(req.body());
  //                auto batch_items = input.value("batch", json::array());

  //                return json{
  //                    {"job_id", "batch_" +
  //                    std::to_string(std::time(nullptr))},
  //                    {"status", "processing"},
  //                    {"batch_size", batch_items.size()},
  //                    {"estimated_completion", std::time(nullptr) + 60},
  //                    {"message", "Batch inference job queued"}};
  //              })
  //       .route(http::verb::get, "/api/v1/jobs/{id}", [](const auto &req) {
  //         // In a real implementation, you'd extract the {id} from the path
  //         return json{{"job_id", "job_12345"},
  //                     {"status", "completed"},
  //                     {"progress", 100},
  //                     {"result_url", "/api/v1/results/job_12345"},
  //                     {"created_at", std::time(nullptr) - 120},
  //                     {"completed_at", std::time(nullptr)}};
  //       });

  //   std::cout << "\n🚀 Surgengine server is ready!" << std::endl;
  //   std::cout << "📋 Available endpoints:" << std::endl;
  //   std::cout << "   GET  http://localhost:8080/          - API overview"
  //             << std::endl;
  //   std::cout << "   GET  http://localhost:8080/heartbeat - Liveness check"
  //             << std::endl;
  //   std::cout << "   GET  http://localhost:8080/health    - Health status"
  //             << std::endl;
  //   std::cout << "   GET  http://localhost:8080/ready     - Readiness check"
  //             << std::endl;
  //   std::cout << "   GET  http://localhost:8080/metrics   - Performance
  //   metrics"
  //             << std::endl;
  //   std::cout << "   POST http://localhost:8080/api/v1/inference/predict -
  //   Run "
  //                "inference"
  //             << std::endl;
  //   std::cout << "\n" << std::endl;

  //   // Start the server (this blocks)
  //   server.run();

  // } catch (const std::exception &e) {
  //   std::cerr << "Server error: " << e.what() << std::endl;
  //   return 1;
  // }

  // return 0;
}
