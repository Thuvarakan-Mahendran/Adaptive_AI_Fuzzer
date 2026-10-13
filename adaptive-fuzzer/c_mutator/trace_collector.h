/*
 * trace_collector.h – Execution trace recording & storage for DL Seed Optimizer.
 *
 * Implements trace buffering and on-disk serialization to /dev/shm.
 * Binary format matches dl_seed_optimizer.trace_reader:
 *   [uint32_t seed_id][uint32_t trace_len][uint16_t edge_ids...]
 */

#ifndef TRACE_COLLECTOR_H
#define TRACE_COLLECTOR_H

#include <stdint.h>
#include <stddef.h>

#define TRACE_DIR "/dev/shm/afl_dl_traces"
#define TRACE_INITIAL_CAPACITY 4096

typedef struct {
    uint32_t  seed_id;
    uint16_t *edges;        /* Dynamic buffer of edge IDs */
    uint32_t  len;          /* Number of recorded edge entries */
    uint32_t  capacity;     /* Allocated capacity in entries */
} trace_buffer_t;

/* Initialize trace storage directory in /dev/shm */
int trace_collector_init(void);

/* Initialize or reset a trace buffer for a new seed */
void trace_begin(trace_buffer_t *buf, uint32_t seed_id);

/* Append an edge ID to the trace */
void trace_append_edge(trace_buffer_t *buf, uint16_t edge_id);

/* Extract hit edges from AFL++ bitmap into trace buffer (Phase 1a) */
uint32_t trace_extract_from_bitmap(trace_buffer_t *buf,
                                   const uint8_t *trace_bits,
                                   size_t bitmap_size);

/* Flush the trace buffer to /dev/shm/afl_dl_traces/<seed_id>.trace.
 * Writes output path into out_path (if non-NULL). */
int trace_flush(const trace_buffer_t *buf, char *out_path, size_t out_path_size);

/* Release trace buffer memory */
void trace_free(trace_buffer_t *buf);

#endif /* TRACE_COLLECTOR_H */
