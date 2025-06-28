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
#include <server/listener.hpp>
#include <server/request_handler.hpp>

namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

class Server {
  net::io_context ioc_;
  std::shared_ptr<RequestHandler> handler_;
  std::shared_ptr<Listener> listener_;
  bool initialized_;

public:
  explicit Server(unsigned short port, int threads = 1)
      : ioc_(threads), initialized_(false) {

    handler_ = std::make_shared<RequestHandler>();

    auto const address = net::ip::make_address("0.0.0.0");
    listener_ = std::make_shared<Listener>(ioc_, tcp::endpoint{address, port},
                                           handler_);

    initialized_ = true;
  }

  // Allow customizing routes before starting
  RequestHandler &routes() { return *handler_; }

  void run() {
    if (!initialized_) {
      std::cerr << "Server not properly initialized" << std::endl;
      return;
    }

    std::cout << "Starting server..." << std::endl;
    listener_->run();
    ioc_.run();
  }

  void stop() { ioc_.stop(); }

  bool is_initialized() const { return initialized_; }
};
