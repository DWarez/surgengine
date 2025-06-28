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
#include <iostream>
#include <nlohmann/json.hpp>
#include <server/server.hpp>

int main() {
  std::cout << "Starting Surgengine server..." << std::endl;

  try {
    Server server(8080);

    if (!server.is_initialized()) {
      std::cerr << "Failed to initialize server" << std::endl;
      return -1;
    }

    server.routes()
        .route(http::verb::get, "/api/inference",
               [](const auto &req) {
                 return json{{"message", "Inference endpoint"},
                             {"available_models",
                              json::array({"gpt", "bert", "resnet"})}};
               })
        .route(http::verb::post, "/api/inference",
               [](const auto &req) {
                 auto input = json::parse(req.body());

                 // Simulate inference processing
                 std::string model = input.value("model", "default");
                 auto data = input.value("data", "");

                 return json{{"result", "inference_complete"},
                             {"model_used", model},
                             {"processed_data", data},
                             {"confidence", 0.95}};
               })
        .route(http::verb::get, "/api/models",
               [](const auto &req) {
                 return json{
                     {"models", json::array({{{"name", "gpt"},
                                              {"type", "language"},
                                              {"status", "ready"}},
                                             {{"name", "bert"},
                                              {"type", "embedding"},
                                              {"status", "ready"}},
                                             {{"name", "resnet"},
                                              {"type", "vision"},
                                              {"status", "loading"}}})}};
               })
        .route(http::verb::post, "/api/models/load", [](const auto &req) {
          auto request_data = json::parse(req.body());
          std::string model_name = request_data.value("model", "unknown");

          return json{{"message", "Loading model: " + model_name},
                      {"estimated_time", "30 seconds"}};
        });

    server.run();

  } catch (const std::exception &e) {
    std::cerr << "Server error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
