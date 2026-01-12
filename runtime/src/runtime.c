#include "cool_runtime.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

// --- Memory Management ---
void* cs_alloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
    return ptr;
}

void cs_free(void* ptr) {
    free(ptr);
}

// --- Task Management (Simple Pthread Wrapper for now) ---
typedef struct {
    cs_task_fn func;
    void* data;
} task_wrapper_t;

static void* task_entry(void* arg) {
    task_wrapper_t* task = (task_wrapper_t*)arg;
    task->func(task->data);
    free(task);
    return NULL;
}

void cs_spawn(cs_task_fn func, void* data) {
    task_wrapper_t* task = (task_wrapper_t*)malloc(sizeof(task_wrapper_t));
    task->func = func;
    task->data = data;

    pthread_t thread;
    // Detached thread so we don't need to join
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    
    if (pthread_create(&thread, &attr, task_entry, task) != 0) {
        fprintf(stderr, "Failed to spawn task\n");
        free(task);
        exit(1);
    }
    pthread_attr_destroy(&attr);
}

// --- Channel Primitives ---
struct cs_channel {
    void** buffer;
    size_t capacity;
    size_t size;
    size_t head;
    size_t tail;
    
    pthread_mutex_t lock;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
    int closed;
};

cs_channel_t* cs_chan_create(size_t capacity) {
    cs_channel_t* chan = (cs_channel_t*)malloc(sizeof(cs_channel_t));
    chan->buffer = (void**)malloc(sizeof(void*) * capacity);
    chan->capacity = capacity;
    chan->size = 0;
    chan->head = 0;
    chan->tail = 0;
    chan->closed = 0;

    pthread_mutex_init(&chan->lock, NULL);
    pthread_cond_init(&chan->not_full, NULL);
    pthread_cond_init(&chan->not_empty, NULL);
    
    return chan;
}

void cs_chan_send(cs_channel_t* chan, void* data) {
    pthread_mutex_lock(&chan->lock);
    
    while (chan->size == chan->capacity && !chan->closed) {
        pthread_cond_wait(&chan->not_full, &chan->lock);
    }

    if (chan->closed) {
        pthread_mutex_unlock(&chan->lock);
        // In a real system, we might return a Result or panic
        fprintf(stderr, "Panic: Send on closed channel\n");
        exit(1);
    }

    chan->buffer[chan->tail] = data;
    chan->tail = (chan->tail + 1) % chan->capacity;
    chan->size++;

    pthread_cond_signal(&chan->not_empty);
    pthread_mutex_unlock(&chan->lock);
}

void* cs_chan_receive(cs_channel_t* chan) {
    pthread_mutex_lock(&chan->lock);
    
    while (chan->size == 0 && !chan->closed) {
        pthread_cond_wait(&chan->not_empty, &chan->lock);
    }

    if (chan->size == 0 && chan->closed) {
        pthread_mutex_unlock(&chan->lock);
        return NULL;
    }

    void* data = chan->buffer[chan->head];
    chan->head = (chan->head + 1) % chan->capacity;
    chan->size--;

    pthread_cond_signal(&chan->not_full);
    pthread_mutex_unlock(&chan->lock);
    
    return data;
}

void cs_chan_close(cs_channel_t* chan) {
    pthread_mutex_lock(&chan->lock);
    chan->closed = 1;
    pthread_cond_broadcast(&chan->not_empty);
    pthread_cond_broadcast(&chan->not_full);
    pthread_mutex_unlock(&chan->lock);
}

// --- String Helpers ---
void cs_print_str(cs_string_t s) {
    // printf("String(len=%ld): ", s.len);
    fwrite(s.ptr, 1, s.len, stdout);
    printf("\n");
}

void cs_print_int(int32_t i) {
    printf("%d\n", i);
}

void cs_sleep(int32_t ms) {
    usleep(ms * 1000);
}
