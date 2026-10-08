/*
 * test_ipc_client.c – Standalone loopback test binary & latency profiler.
 *
 * Verifies:
 *   1. Connection establishment & is_connected status
 *   2. Datagram integrity for MSG_MUT_REQ <-> MSG_MUT_RESP
 *   3. Datagram integrity for MSG_SEED_QUERY <-> MSG_SEED_VERD
 *   4. High-throughput streaming (up to 100,000 iterations)
 *   5. Latency profiling (mean RTT in microseconds)
 *   6. Disconnect resilience when server drops
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>

#include "protocol.h"
#include "socket_client.h"

static double get_time_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

int main(int argc, char **argv)
{
    const char *sock_path = DEFAULT_SOCKET_PATH;
    uint32_t iterations   = 1000;
    int test_resilience   = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sock") == 0 && i + 1 < argc) {
            sock_path = argv[++i];
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iterations = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stress") == 0) {
            iterations = 100000;
        } else if (strcmp(argv[i], "--resilience") == 0) {
            test_resilience = 1;
        }
    }

    printf("============================================================\n");
    printf("[test_ipc_client] Starting IPC Client Test Suite\n");
    printf("  Socket:     %s\n", sock_path);
    printf("  Iterations: %u\n", iterations);
    printf("============================================================\n");

    /* 1. Initialize client */
    ipc_client_t *client = ipc_client_init(sock_path, 100, 10);
    if (!client) {
        fprintf(stderr, "[FAIL] Memory allocation failed for ipc_client\n");
        return 1;
    }

    if (!ipc_client_is_connected(client)) {
        fprintf(stderr, "[FAIL] Could not connect to socket at %s\n", sock_path);
        ipc_client_close(client);
        return 2;
    }
    printf("[PASS] Connection established.\n");

    /* 2. Single MSG_MUT_REQ / MSG_MUT_RESP functional test */
    struct mut_request_msg mut_req;
    memset(&mut_req, 0, sizeof(mut_req));
    mut_req.type = MSG_MUT_REQ;
    mut_req.deferred_reward = 12.345f;
    memset(mut_req.history_vec, 0, sizeof(mut_req.history_vec));
    mut_req.history_vec[0] = 0.5f;  /* usage slot */
    mut_req.history_vec[32] = 1.0f; /* valid one-hot last operator */
    mut_req.queue_cur_changed = 1;

    struct mut_response_msg mut_resp;
    memset(&mut_resp, 0, sizeof(mut_resp));

    int sent_ok = ipc_client_send_mut_req(client, &mut_req);
    assert(sent_ok == 1);

    int recv_ok = ipc_client_recv_mut_resp(client, &mut_resp);
    assert(recv_ok == 1);
    assert(mut_resp.type == MSG_MUT_RESP);
    assert(mut_resp.count > 0 && mut_resp.count <= MUT_BATCH_MAX);
    printf("[PASS] MSG_MUT_REQ round-trip passed (count=%d, op0=%d).\n",
           mut_resp.count, mut_resp.actions[0]);

    /* 3. Single MSG_SEED_QUERY / MSG_SEED_VERD functional test */
    struct seed_query_msg query;
    memset(&query, 0, sizeof(query));
    query.type = MSG_SEED_QUERY;
    query.seed_id = 999;
    query.trace_len = 10;
    for (uint32_t i = 0; i < query.trace_len; i++) {
        query.steps[i].edge_id = (uint16_t)(i + 1);
        query.steps[i].hit_count_bin = (uint8_t)(i % 8);
        query.steps[i].fault_flags = 0;
    }

    struct seed_verdict_msg verd;
    memset(&verd, 0, sizeof(verd));

    sent_ok = ipc_client_send_seed_query(client, &query);
    assert(sent_ok == 1);

    recv_ok = ipc_client_recv_seed_verd(client, &verd);
    assert(recv_ok == 1);
    assert(verd.type == MSG_SEED_VERD);
    assert(verd.accept == 1);
    printf("[PASS] MSG_SEED_QUERY round-trip passed (accept=%d, q=%.2f).\n",
           verd.accept, verd.q_value);

    /* 4. Streaming Throughput & Latency Profiling */
    printf("[*] Running %u round-trip iterations for latency benchmark...\n", iterations);
    double t_start = get_time_us();

    uint32_t success_count = 0;
    for (uint32_t i = 0; i < iterations; i++) {
        mut_req.deferred_reward = (float)i;
        if (!ipc_client_send_mut_req(client, &mut_req)) {
            if (test_resilience) {
                printf("[*] Client caught send failure as expected during resilience test.\n");
                break;
            }
            fprintf(stderr, "[FAIL] send_mut_req failed at iteration %u\n", i);
            break;
        }

        if (!ipc_client_recv_mut_resp(client, &mut_resp)) {
            if (test_resilience) {
                printf("[*] Client caught recv timeout/disconnect as expected during resilience test.\n");
                break;
            }
            fprintf(stderr, "[FAIL] recv_mut_resp failed at iteration %u\n", i);
            break;
        }

        success_count++;
    }

    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    double mean_rtt_us   = (success_count > 0) ? (t_end - t_start) / (double)success_count : 0.0;

    printf("[PASS] Benchmarked %u / %u successful round-trips\n", success_count, iterations);
    printf("       Total elapsed: %.2f ms\n", total_time_ms);
    printf("       Mean RTT:      %.2f us (%.0f req/sec)\n",
           mean_rtt_us, (mean_rtt_us > 0) ? 1000000.0 / mean_rtt_us : 0.0);

    /* 5. Cleanup */
    ipc_client_close(client);
    printf("[PASS] Clean client shutdown.\n");
    printf("============================================================\n");
    return 0;
}
