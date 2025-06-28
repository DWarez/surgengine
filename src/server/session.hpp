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
#include <nlohmann/json.hpp>
#include <server/request_handler.hpp>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

class Session : public std::enable_shared_from_this<Session> {
  tcp::socket socket_;
  beast::flat_buffer buffer_;
  http::request<http::string_body> request_;
  std::shared_ptr<RequestHandler> handler_;

public:
  explicit Session(tcp::socket socket, std::shared_ptr<RequestHandler> handler)
      : socket_(std::move(socket)), handler_(std::move(handler)) {}

  void run() { do_read(); }

private:
  void do_read() {
    auto self(shared_from_this());
    http::async_read(socket_, buffer_, request_,
                     [this, self](beast::error_code ec, std::size_t) {
                       if (!ec) {
                         do_write(handler_->handle_request(request_));
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

          if (!res->keep_alive()) {
            beast::error_code shutdown_ec;
            socket_.shutdown(tcp::socket::shutdown_send, shutdown_ec); // NOLINT
            if (shutdown_ec) {
              std::cerr << "Socket shutdown error: " << shutdown_ec.message()
                        << std::endl;
            }
          } else {
            do_read();
          }
        });
  }
};