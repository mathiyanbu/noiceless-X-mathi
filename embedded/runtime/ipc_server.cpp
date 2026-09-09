#include "ipc_server.hpp"
#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>
#include <chrono>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
#define IS_VALID_SOCKET(s) ((s) != INVALID_SOCKET)
#define CLOSE_SOCKET(s) closesocket(s)
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
using socket_t = int;
#define IS_VALID_SOCKET(s) ((s) >= 0)
#define CLOSE_SOCKET(s) close(s)
#endif

namespace noiselessx::runtime {

IpcServer::IpcServer(uint16_t port, const std::string& unix_path)
    : port_(port), unix_path_(unix_path)
{
}

IpcServer::~IpcServer() {
    stop();
}

bool IpcServer::start() {
    if (is_running_.load()) {
        return true;
    }

    should_stop_.store(false);
    is_running_.store(true);

    server_thread_ = std::thread(&IpcServer::listener_thread_loop, this);
    std::cout << "[IpcServer] IPC Server active (Port: " << port_ 
              << ", UDS: " << unix_path_ << ")\n";
    return true;
}

void IpcServer::stop() {
    if (!is_running_.load()) {
        return;
    }

    should_stop_.store(true);
    if (server_thread_.joinable()) {
        server_thread_.join();
    }

#if !defined(_WIN32)
    unlink(unix_path_.c_str());
#endif

    is_running_.store(false);
    std::cout << "[IpcServer] IPC Server stopped cleanly.\n";
}

std::string IpcServer::dispatch_command(const std::string& raw_request) {
    std::string cmd;
    std::string body;

    // Extract command from simple string or JSON
    auto cmd_pos = raw_request.find("\"command\":");
    if (cmd_pos != std::string::npos) {
        auto quote1 = raw_request.find('"', cmd_pos + 10);
        if (quote1 != std::string::npos) {
            auto quote2 = raw_request.find('"', quote1 + 1);
            if (quote2 != std::string::npos) {
                cmd = raw_request.substr(quote1 + 1, quote2 - quote1 - 1);
            }
        }
    } else {
        std::istringstream iss(raw_request);
        iss >> cmd;
    }

    if (cmd.empty()) {
        cmd = "ping";
    }

    if (handler_) {
        return handler_(cmd, raw_request);
    }

    // Default internal handler if no custom handler set
    if (cmd == "ping") {
        return "{\"status\":\"ok\",\"runtime\":\"ready\"}\n";
    }

    return "{\"status\":\"ok\",\"message\":\"command acknowledged\"}\n";
}

void IpcServer::listener_thread_loop() {
#if defined(_WIN32)
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif

    socket_t listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (!IS_VALID_SOCKET(listen_fd)) {
        is_running_.store(false);
        return;
    }

    int opt = 1;
#if defined(_WIN32)
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // 127.0.0.1
    server_addr.sin_port = htons(port_);

    if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        CLOSE_SOCKET(listen_fd);
        is_running_.store(false);
        return;
    }

    if (listen(listen_fd, 5) < 0) {
        CLOSE_SOCKET(listen_fd);
        is_running_.store(false);
        return;
    }

    while (!should_stop_.load(std::memory_order_relaxed)) {
        // Use select with timeout so thread can exit promptly
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(listen_fd, &read_fds);

        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 200000; // 200 ms timeout

        int sel = select(static_cast<int>(listen_fd + 1), &read_fds, nullptr, nullptr, &tv);
        if (sel <= 0) {
            continue;
        }

        sockaddr_in client_addr{};
#if defined(_WIN32)
        int client_len = sizeof(client_addr);
#else
        socklen_t client_len = sizeof(client_addr);
#endif
        socket_t client_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
        if (!IS_VALID_SOCKET(client_fd)) {
            continue;
        }

        // Read request from client
        char buffer[4096];
        int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            std::string response = dispatch_command(std::string(buffer));
            if (response.back() != '\n') {
                response += '\n';
            }
            send(client_fd, response.c_str(), static_cast<int>(response.length()), 0);
        }

        CLOSE_SOCKET(client_fd);
    }

    CLOSE_SOCKET(listen_fd);

#if defined(_WIN32)
    WSACleanup();
#endif
}

} // namespace noiselessx::runtime
