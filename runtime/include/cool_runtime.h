#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- Memory Management ---
void* cs_alloc(size_t size);
void cs_free(void* ptr);

// --- Task & Thread Management ---
typedef void (*cs_task_fn)(void*);
void cs_spawn(cs_task_fn func, void* data);

// --- Channel Primitives ---
typedef struct cs_channel cs_channel_t;

cs_channel_t* cs_chan_create(size_t capacity);
void cs_chan_send(cs_channel_t* chan, void* data);
void* cs_chan_receive(cs_channel_t* chan);
void cs_chan_close(cs_channel_t* chan);

// --- String Helpers ---
typedef struct {
    char* ptr;
    int64_t len;
} cs_string_t;

void cs_print_str(cs_string_t s);
void cs_print_int(int32_t i); // Added for basic testing
void cs_sleep(int32_t ms);

#ifdef __cplusplus
}
#endif
