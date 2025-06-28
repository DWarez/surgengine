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
#include <boost/beast/http/verb.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

#pragma once
class RequestHandler {
public:
  using Handler = std::function<json(const http::request<http::string_body> &)>;

private:
  std::unordered_map<std::string, std::unordered_map<http::verb, Handler>>
      routes_;

  http::response<http::string_body>
  make_json_response(const http::request<http::string_body> &request,
                     const json &body, http::status status = http::status::ok) {
    http::response<http::string_body> response{status, request.version()};
    response.set(http::field::server, "Surgengine");
    response.set(http::field::content_type, "application/json");
    response.keep_alive(request.keep_alive());
    response.body() = body.dump();
    response.prepare_payload();
    return response;
  }

  http::response<http::string_body>
  make_error_response(const http::request<http::string_body> &request,
                      http::status status, const std::string &message) {
    json error_body = {{"error", message}};
    return make_json_response(request, error_body, status);
  }

public:
  RequestHandler() { setup_routes(); }

  RequestHandler &route(http::verb method, const std::string &path,
                        Handler handler) {
    routes_[path][method] = std::move(handler);
    return *this;
  }

  http::response<http::string_body>
  handle_request(const http::request<http::string_body> &request) {
    try {
      auto path_it = routes_.find(std::string(request.target()));
      if (path_it == routes_.end()) {
        return make_error_response(request, http::status::not_found,
                                   "Route not found");
      }

      auto method_it = path_it->second.find(request.method());
      if (method_it == path_it->second.end()) {
        return make_error_response(request, http::status::method_not_allowed,
                                   "Method not allowed");
      }

      json response_body = method_it->second(request);
      return make_json_response(request, response_body);

    } catch (const json::parse_error &e) {
      return make_error_response(request, http::status::bad_request,
                                 "Invalid JSON: " + std::string(e.what()));
    } catch (const std::exception &e) {
      return make_error_response(request, http::status::internal_server_error,
                                 "Internal error: " + std::string(e.what()));
    }
  }

private:
  void setup_routes() {
    route(http::verb::get, "/api/data", [](const auto &req) {
      return json{{"message", "This is a GET request"}};
    });

    route(http::verb::post, "/api/data", [](const auto &req) {
      auto request_json = json::parse(req.body());
      return json{{"message", "Received: " + request_json.dump()}};
    });

    route(http::verb::put, "/api/data", [](const auto &req) {
      auto request_json = json::parse(req.body());
      return json{{"message", "Updated: " + request_json.dump()}};
    });

    route(http::verb::delete_, "/api/data", [](const auto &req) {
      return json{{"message", "Resource deleted"}};
    });

    route(http::verb::get, "/", [](const auto &req) {
      return json{{"message", "Welcome to the Surgengine inference engine"}};
    });
  }
};