#pragma once
#include "cstdint"

using SOCKET = uintptr_t;
#define INVALID_SOCKET (static_cast<SOCKET>(~0))
#define SOCKET_ERROR (-1)
#define AF_INET 2
#define SOCK_STREAM 1
#define SOL_SOCKET 0xffff
#define SO_REUSEADDR 0x0004
#define INADDR_ANY 0x00000000

struct WSADATA {
    uint16_t wVersion;
    uint16_t wHighVersion;
    char szDescription[257];
    char szSystemStatus[129];
};

inline int WSAStartup(uint16_t, WSADATA*) { return 0; }
inline int WSACleanup() { return 0; }
inline int closesocket(SOCKET) { return 0; }
#define MAKEWORD(a, b) ((uint16_t)(((uint8_t)(((uintptr_t)(a)) & 0xff)) | ((uint16_t)((uint8_t)(((uintptr_t)(b)) & 0xff))) << 8))

struct in_addr {
    uint32_t s_addr;
};

struct sockaddr_in {
    int16_t sin_family;
    uint16_t sin_port;
    in_addr sin_addr;
    char sin_zero[8];
};

struct sockaddr {
    uint16_t sa_family;
    char sa_data[14];
};

inline SOCKET socket(int, int, int) { return INVALID_SOCKET; }
inline int setsockopt(SOCKET, int, int, const char*, int) { return 0; }
inline int bind(SOCKET, const sockaddr*, int) { return 0; }
inline int listen(SOCKET, int) { return 0; }
inline SOCKET accept(SOCKET, sockaddr*, int*) { return INVALID_SOCKET; }
inline int send(SOCKET, const char*, int, int) { return 0; }
inline int recv(SOCKET, char*, int, int) { return 0; }
inline uint16_t htons(uint16_t v) { return v; }
inline unsigned long inet_addr(const char*) { return 0; }
