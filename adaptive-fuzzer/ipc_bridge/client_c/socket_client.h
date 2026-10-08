/*
 * socket_client.h – C IPC client interface for AFL++ AI bridge.
 *
 * Provides a robust, non-leaking AF_UNIX SOCK_SEQPACKET client abstraction.
 * All functions return 1 on success, 0 on failure or timeout.
 */

#ifndef SOCKET_CLIENT_H
#define SOCKET_CLIENT_H

#include <stdint.h>
#include <stddef.h>
#include "protocol.h"

/* Opaque IPC client handle */
typedef struct ipc_client ipc_client_t;

/**
 * ipc_client_init – Create and connect an IPC client socket.
 *
 * @param socket_path   Path to the UNIX domain socket (e.g., DEFAULT_SOCKET_PATH).
 *                      If NULL, DEFAULT_SOCKET_PATH is used.
 * @param timeout_ms    Receive timeout in milliseconds (SO_RCVTIMEO). If 0, default 100ms.
 * @param max_retries   Maximum connection attempts before giving up (backoff ~10ms each).
 *
 * @return  Allocated ipc_client_t pointer, or NULL on memory allocation failure.
 *          Note: If connection fails after max_retries, the handle is still returned
 *          with is_connected() == 0 so fallback policies work cleanly without NULL checks.
 */
ipc_client_t *ipc_client_init(const char *socket_path, uint32_t timeout_ms, uint32_t max_retries);

/**
 * ipc_client_is_connected – Check if the client currently holds an active connection.
 *
 * @return 1 if connected, 0 otherwise.
 */
int ipc_client_is_connected(const ipc_client_t *client);

/**
 * ipc_client_reconnect – Attempt to reconnect an inactive or disconnected client.
 *
 * @return 1 on successful reconnection, 0 on failure.
 */
int ipc_client_reconnect(ipc_client_t *client);

/**
 * ipc_client_send_mut_req – Transmit a MSG_MUT_REQ datagram to the server.
 * Uses MSG_NOSIGNAL to prevent SIGPIPE.
 *
 * @return 1 on success, 0 on failure.
 */
int ipc_client_send_mut_req(ipc_client_t *client, const struct mut_request_msg *req);

/**
 * ipc_client_recv_mut_resp – Receive a MSG_MUT_RESP datagram from the server.
 * Respects SO_RCVTIMEO.
 *
 * @return 1 on success, 0 on failure or timeout.
 */
int ipc_client_recv_mut_resp(ipc_client_t *client, struct mut_response_msg *resp);

/**
 * ipc_client_send_seed_query – Transmit a MSG_SEED_QUERY datagram to the server.
 * Uses MSG_NOSIGNAL to prevent SIGPIPE.
 *
 * @return 1 on success, 0 on failure.
 */
int ipc_client_send_seed_query(ipc_client_t *client, const struct seed_query_msg *query);

/**
 * ipc_client_recv_seed_verd – Receive a MSG_SEED_VERD datagram from the server.
 * Respects SO_RCVTIMEO.
 *
 * @return 1 on success, 0 on failure or timeout.
 */
int ipc_client_recv_seed_verd(ipc_client_t *client, struct seed_verdict_msg *verd);

/**
 * ipc_client_close – Close socket connection and free client resources.
 */
void ipc_client_close(ipc_client_t *client);

#endif /* SOCKET_CLIENT_H */
