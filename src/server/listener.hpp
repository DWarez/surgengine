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
#include <server/session.hpp>

class Listener : public std::enable_shared_from_this<Listener> {
  net::io_context &ioc_;
  tcp::acceptor acceptor_;
  std::shared_ptr<RequestHandler> handler_;

public:
  Listener(net::io_context &ioc, tcp::endpoint endpoint,
           std::shared_ptr<RequestHandler> handler)
      : ioc_(ioc), acceptor_(net::make_strand(ioc)),
        handler_(std::move(handler)) {

    beast::error_code ec;

    acceptor_.open(endpoint.protocol(), ec); // NOLINT
    if (ec) {
      std::cerr << "Failed to open acceptor: " << ec.message() << std::endl;
      return;
    }

    acceptor_.set_option(net::socket_base::reuse_address(true), ec); // NOLINT
    if (ec) {
      std::cerr << "Failed to set socket option: " << ec.message() << std::endl;
      return;
    }

    acceptor_.bind(endpoint, ec); // NOLINT
    if (ec) {
      std::cerr << "Failed to bind endpoint: " << ec.message() << std::endl;
      return;
    }

    acceptor_.listen(net::socket_base::max_listen_connections, ec); // NOLINT
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
        std::make_shared<Session>(std::move(socket), handler_)->run();
      } else {
        std::cerr << "Accept error: " << ec.message() << std::endl;
      }

      do_accept();
    });
  }
};
