#include <server/server.hpp>

int main() {
  std::cout << "Starting server" << std::endl;
  try {
    auto const address = net::ip::make_address("0.0.0.0");
    unsigned short port = 8080;

    net::io_context ioc{1};

    std::cout << "Creating listener on port " << port << std::endl;
    auto listener =
        std::make_shared<Listener>(ioc, tcp::endpoint{address, port});
    listener->run();

    std::cout << "Starting io_context..." << std::endl;
    ioc.run();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}