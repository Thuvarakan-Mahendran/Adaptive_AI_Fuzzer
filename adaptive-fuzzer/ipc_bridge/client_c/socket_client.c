/*
 * socket_client.c – Hardened AF_UNIX SOCK_SEQPACKET IPC client implementation.
 *
 * Implements:
 *   - FD_CLOEXEC to prevent leaking the socket into AFL++ child forkservers.
 *   - SO_RCVTIMEO for bounded, deterministic latency on receive.
 *   - MSG_NOSIGNAL on all transmissions to prevent SIGPIPE crashes.
 *   - Safe retry backoff to survive startup launch races.
 *   - Automatic disconnect detection and error contract (1=ok, 0=fallback).
 */

#include "socket_client.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* Default settings */
#define DEFAULT_TIMEOUT_MS  100
#define BACKOFF_SLEEP_US    10000   /* 10 ms backoff */

struct ipc_client {
    int      sock_fd;
    char     socket_path[sizeof(((struct sockaddr_un *)0)->sun_path)];
    uint32_t timeout_ms;
    int      connected;
};

/* Internal helper: attempt to create and connect a socket */
static int do_connect(const char *path, uint32_t timeout_ms)
{
    int fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (fd < 0)
        return -1;

    /* Ensure child processes spawned by AFL++ forkserver do not inherit this fd */
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
        close(fd);
        return -1;
    }

    /* Apply receive timeout */
    struct timeval tv;
    tv.tv_sec  = (time_t)(timeout_ms / 1000);
    tv.tv_usec = (suseconds_t)((timeout_ms % 1000) * 1000);
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        close(fd);
        return -1;
    }

    /* Connect to UNIX domain socket */
    struct sockaddr_un sun;
    memset(&sun, 0, sizeof(sun));
    sun.sun_family = AF_UNIX;
    strncpy(sun.sun_path, path, sizeof(sun.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&sun, sizeof(sun)) < 0) {
        close(fd);
        return -1;
    }

    return fd;
}

ipc_client_t *ipc_client_init(const char *socket_path, uint32_t timeout_ms, uint32_t max_retries)
{
    ipc_client_t *client = (ipc_client_t *)calloc(1, sizeof(ipc_client_t));
    if (!client)
        return NULL;

    const char *path = socket_path ? socket_path : DEFAULT_SOCKET_PATH;
    strncpy(client->socket_path, path, sizeof(client->socket_path) - 1);
    client->timeout_ms = timeout_ms ? timeout_ms : DEFAULT_TIMEOUT_MS;
    client->sock_fd    = -1;
    client->connected  = 0;

    uint32_t retries = (max_retries == 0) ? 1 : max_retries;
    for (uint32_t i = 0; i < retries; i++) {
        client->sock_fd = do_connect(client->socket_path, client->timeout_ms);
        if (client->sock_fd >= 0) {
            client->connected = 1;
            break;
        }
        if (i + 1 < retries) {
            usleep(BACKOFF_SLEEP_US);
        }
    }

    return client;
}

int ipc_client_is_connected(const ipc_client_t *client)
{
    return (client && client->connected && client->sock_fd >= 0) ? 1 : 0;
}

int ipc_client_reconnect(ipc_client_t *client)
{
    if (!client)
        return 0;

    if (client->connected && client->sock_fd >= 0)
        return 1;

    if (client->sock_fd >= 0) {
        close(client->sock_fd);
        client->sock_fd = -1;
    }
    client->connected = 0;

    client->sock_fd = do_connect(client->socket_path, client->timeout_ms);
    if (client->sock_fd >= 0) {
        client->connected = 1;
        return 1;
    }

    return 0;
}

static void handle_transport_error(ipc_client_t *client)
{
    if (client && client->sock_fd >= 0) {
        close(client->sock_fd);
        client->sock_fd = -1;
        client->connected = 0;
    }
}

int ipc_client_send_mut_req(ipc_client_t *client, const struct mut_request_msg *req)
{
    if (!client || !client->connected || client->sock_fd < 0 || !req)
        return 0;

    ssize_t sent = send(client->sock_fd, req, sizeof(*req), MSG_NOSIGNAL);
    if (sent != (ssize_t)sizeof(*req)) {
        if (sent < 0 && (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)) {
            handle_transport_error(client);
        }
        return 0;
    }

    return 1;
}

int ipc_client_recv_mut_resp(ipc_client_t *client, struct mut_response_msg *resp)
{
    if (!client || !client->connected || client->sock_fd < 0 || !resp)
        return 0;

    ssize_t recvd = recv(client->sock_fd, resp, sizeof(*resp), 0);
    if (recvd <= 0) {
        if (recvd == 0 || (recvd < 0 && (errno == ECONNRESET || errno == ENOTCONN))) {
            handle_transport_error(client);
        }
        return 0;
    }

    if (recvd != (ssize_t)sizeof(*resp) || resp->type != MSG_MUT_RESP)
        return 0;

    return 1;
}

int ipc_client_send_seed_query(ipc_client_t *client, const struct seed_query_msg *query)
{
    if (!client || !client->connected || client->sock_fd < 0 || !query)
        return 0;

    ssize_t sent = send(client->sock_fd, query, sizeof(*query), MSG_NOSIGNAL);
    if (sent != (ssize_t)sizeof(*query)) {
        if (sent < 0 && (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)) {
            handle_transport_error(client);
        }
        return 0;
    }

    return 1;
}

int ipc_client_recv_seed_verd(ipc_client_t *client, struct seed_verdict_msg *verd)
{
    if (!client || !client->connected || client->sock_fd < 0 || !verd)
        return 0;

    ssize_t recvd = recv(client->sock_fd, verd, sizeof(*verd), 0);
    if (recvd <= 0) {
        if (recvd == 0 || (recvd < 0 && (errno == ECONNRESET || errno == ENOTCONN))) {
            handle_transport_error(client);
        }
        return 0;
    }

    if (recvd != (ssize_t)sizeof(*verd) || verd->type != MSG_SEED_VERD)
        return 0;

    return 1;
}

void ipc_client_close(ipc_client_t *client)
{
    if (!client)
        return;

    if (client->sock_fd >= 0) {
        close(client->sock_fd);
        client->sock_fd = -1;
    }
    client->connected = 0;
    free(client);
}
