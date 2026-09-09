#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <memory>

namespace noiselessx {
namespace runtime {

/**
 * @brief Lightweight Local IPC Server for Real-Time Runtime Control and Telemetry.
 * 
 * Exposes runtime status and control endpoints to the FastAPI backend over:
 * 1. Unix Domain Socket (/tmp/noiselessx.sock on Linux)
 * 2. Localhost TCP loopback (127.0.0.1:9099)
 * 
 * Runs in a detached low-priority background thread so it NEVER blocks or interrupts
 * the real-time audio threads (SCHED_FIFO).
 */
class IpcServer {
public:
    using RequestHandler = std::function<std::string(const std::string& command, const std::string& body)>;

    explicit IpcServer(uint16_t port = 9099, const std::string& unix_path = "/tmp/noiselessx.sock");
    ~IpcServer();

    // Non-copyable, non-movable
    IpcServer(const IpcServer&) = delete;
    IpcServer& operator=(const IpcServer&) = delete;

    /**
     * @brief Register the callback invoked for incoming command messages.
     */
    void set_request_handler(RequestHandler handler) {
        handler_ = std::move(handler);
    }

    /**
     * @brief Start background IPC listener thread.
     */
    bool start();

    /**
     * @brief Stop IPC listener thread and clean up socket files.
     */
    void stop();

    [[nodiscard]] bool is_running() const noexcept {
        return is_running_.load();
    }

    [[nodiscard]] const std::string& get_socket_path() const noexcept {
        return unix_path_;
    }

    [[nodiscard]] uint16_t get_port() const noexcept {
        return port_;
    }

private:
    void listener_thread_loop();
    std::string dispatch_command(const std::string& raw_request);

    uint16_t port_{9099};
    std::string unix_path_{"/tmp/noiselessx.sock"};
    std::atomic<bool> is_running_{false};
    std::atomic<bool> should_stop_{false};

    std::thread server_thread_;
    RequestHandler handler_{nullptr};
};

} // namespace runtime
} // namespace noiselessx
