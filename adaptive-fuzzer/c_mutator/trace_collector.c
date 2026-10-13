/*
 * trace_collector.c – Execution trace recording & storage for DL Seed Optimizer.
 */

#include "trace_collector.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

int trace_collector_init(void)
{
    struct stat st;
    if (stat(TRACE_DIR, &st) == 0) {
        if (S_ISDIR(st.st_mode))
            return 1;
    }

    if (mkdir(TRACE_DIR, 0777) < 0 && errno != EEXIST) {
        perror("[trace_collector] mkdir failed for " TRACE_DIR);
        return 0;
    }
    return 1;
}

void trace_begin(trace_buffer_t *buf, uint32_t seed_id)
{
    if (!buf)
        return;

    buf->seed_id = seed_id;
    buf->len     = 0;

    if (!buf->edges) {
        buf->capacity = TRACE_INITIAL_CAPACITY;
        buf->edges    = (uint16_t *)malloc(buf->capacity * sizeof(uint16_t));
        if (!buf->edges) {
            buf->capacity = 0;
            perror("[trace_collector] malloc trace_buffer failed");
        }
    }
}

void trace_append_edge(trace_buffer_t *buf, uint16_t edge_id)
{
    if (!buf)
        return;

    if (buf->len >= buf->capacity) {
        uint32_t new_cap = (buf->capacity == 0) ? TRACE_INITIAL_CAPACITY : buf->capacity * 2;
        uint16_t *new_ptr = (uint16_t *)realloc(buf->edges, new_cap * sizeof(uint16_t));
        if (!new_ptr) {
            perror("[trace_collector] realloc trace_buffer failed");
            return;
        }
        buf->edges    = new_ptr;
        buf->capacity = new_cap;
    }

    buf->edges[buf->len++] = edge_id;
}

uint32_t trace_extract_from_bitmap(trace_buffer_t *buf,
                                   const uint8_t *trace_bits,
                                   size_t bitmap_size)
{
    if (!buf || !trace_bits || bitmap_size == 0)
        return 0;

    uint32_t added = 0;
    for (size_t i = 0; i < bitmap_size; i++) {
        if (trace_bits[i]) {
            trace_append_edge(buf, (uint16_t)(i & 0xFFFF));
            added++;
        }
    }
    return added;
}

int trace_flush(const trace_buffer_t *buf, char *out_path, size_t out_path_size)
{
    if (!buf)
        return 0;

    char path[256];
    snprintf(path, sizeof(path), "%s/%u.trace", TRACE_DIR, buf->seed_id);

    if (out_path && out_path_size > 0) {
        strncpy(out_path, path, out_path_size - 1);
        out_path[out_path_size - 1] = '\0';
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror("[trace_collector] fopen trace file failed");
        return 0;
    }

    uint32_t header[2];
    header[0] = buf->seed_id;
    header[1] = buf->len;

    if (fwrite(header, sizeof(uint32_t), 2, fp) != 2) {
        fclose(fp);
        return 0;
    }

    if (buf->len > 0 && buf->edges) {
        if (fwrite(buf->edges, sizeof(uint16_t), buf->len, fp) != buf->len) {
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}

void trace_free(trace_buffer_t *buf)
{
    if (!buf)
        return;

    if (buf->edges) {
        free(buf->edges);
        buf->edges = NULL;
    }
    buf->len      = 0;
    buf->capacity = 0;
}
