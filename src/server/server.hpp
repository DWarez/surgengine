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
#include <memory>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

inline http::response<http::string_body>
handle_request(http::request<http::string_body> const &request) {
  if (request.method() == http::verb::get) {
    http::response<http::string_body> response{http::status::ok,
                                               request.version()};
    response.set(http::field::server, "Surgengine");
    response.set(http::field::content_type, "text/plain");
    response.keep_alive(request.keep_alive());
    response.body() = "Welcome to the Surgengine inference engine.";
    response.prepare_payload();
    return response;
  }

  // Fixed: properly handle error responses
  http::response<http::string_body> response{http::status::bad_request,
                                             request.version()};
  response.set(http::field::server, "Surgengine");
  response.set(http::field::content_type, "text/plain");
  response.body() = "Bad Request";
  response.prepare_payload();
  return response;
}

class Session : public std::enable_shared_from_this<Session> {
  tcp::socket socket_;
  beast::flat_buffer buffer_;
  http::request<http::string_body> request_;

public:
  explicit Session(tcp::socket socket) : socket_(std::move(socket)) {}

  void run() { do_read(); }

private:
  void do_read() {
    auto self(shared_from_this());
    http::async_read(socket_, buffer_, request_,
                     [this, self](beast::error_code ec, std::size_t) {
                       if (!ec) {
                         do_write(handle_request(request_));
                       } else {
                         std::cerr << "Read error: " << ec.message()
                                   << std::endl;
                       }
                     });
  }

  void do_write(http::response<http::string_body> response) {
    auto self(shared_from_this());
    auto res = std::make_shared<http::response<http::string_body>>(
        std::move(response));
    http::async_write(
        socket_, *res, [this, self, res](beast::error_code ec, std::size_t) {
          if (ec) {
            std::cerr << "Write error: " << ec.message() << std::endl;
          }

          // Only shutdown if we're not keeping the connection alive
          if (!res->keep_alive()) {
            beast::error_code shutdown_ec;
            socket_.shutdown(tcp::socket::shutdown_send, shutdown_ec);
            if (shutdown_ec) {
              std::cerr << "Socket shutdown error: " << shutdown_ec.message()
                        << std::endl;
            }
          } else {
            // Keep connection alive - read next request
            do_read();
          }
        });
  }
};

class Listener : public std::enable_shared_from_this<Listener> {
  net::io_context &ioc_;
  tcp::acceptor acceptor_;

public:
  Listener(net::io_context &ioc, tcp::endpoint endpoint)
      : ioc_(ioc), acceptor_(net::make_strand(ioc)) {
    beast::error_code ec;

    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
      std::cerr << "Failed to open acceptor: " << ec.message() << std::endl;
      return;
    }

    acceptor_.set_option(net::socket_base::reuse_address(true), ec);
    if (ec) {
      std::cerr << "Failed to set socket option: " << ec.message() << std::endl;
      return;
    }

    acceptor_.bind(endpoint, ec);
    if (ec) {
      std::cerr << "Failed to bind endpoint: " << ec.message() << std::endl;
      return;
    }

    acceptor_.listen(net::socket_base::max_listen_connections, ec);
    if (ec) {
      std::cerr << "Failed to listen: " << ec.message() << std::endl;
      return;
    }

    std::cout << "Server listening on " << endpoint << std::endl;
  }

  void run() { do_accept(); }

private:
  void do_accept() {
    acceptor_.async_accept(net::make_strand(ioc_), [this](beast::error_code ec,
                                                          tcp::socket socket) {
      if (!ec) {
        std::cout << "New connection accepted" << std::endl;
        std::make_shared<Session>(std::move(socket))->run();
      } else {
        std::cerr << "Accept error: " << ec.message() << std::endl;
      }

      // Continue accepting new connections
      do_accept();
    });
  }
};