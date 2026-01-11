To ensure that **Coolscript** tasks can communicate safely and efficiently, the C runtime must implement a thread-safe **Channel** that respects the language's ownership model. Since Coolscript ensures that only one task can "own" a piece of data at a time, the channel acts as a secure hand-off point.

The following C implementation uses a **Mutex** and **Condition Variables** to handle blocking, ensuring that `send` waits if the channel is full and `receive` waits if it is empty.

---

## Coolscript Channel Implementation (`runtime.c`)

This implementation handles the transfer of "owned" pointers between the memory spaces of different tasks.

```c
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct cs_channel {
    void** buffer;          // Array of pointers (the "owned" resources)
    size_t capacity;        // Max items in channel
    size_t size;            // Current items in channel
    size_t head;            // Read index
    size_t tail;            // Write index
    
    pthread_mutex_t lock;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;
    int closed;
} cs_channel_t;

// --- Initialize Channel ---
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

// --- Send (Move ownership INTO channel) ---
void cs_chan_send(cs_channel_t* chan, void* data) {
    pthread_mutex_lock(&chan->lock);
    
    while (chan->size == chan->capacity && !chan->closed) {
        pthread_cond_wait(&chan->not_full, &chan->lock);
    }

    if (chan->closed) {
        pthread_mutex_unlock(&chan->lock);
        return; // Or trigger a controlled panic
    }

    // Hand off the pointer to the internal buffer
    chan->buffer[chan->tail] = data;
    chan->tail = (chan->tail + 1) % chan->capacity;
    chan->size++;

    pthread_cond_signal(&chan->not_empty);
    pthread_mutex_unlock(&chan->lock);
}

// --- Receive (Move ownership OUT of channel) ---
void* cs_chan_receive(cs_channel_t* chan) {
    pthread_mutex_lock(&chan->lock);
    
    while (chan->size == 0 && !chan->closed) {
        pthread_cond_wait(&chan->not_empty, &chan->lock);
    }

    if (chan->size == 0 && chan->closed) {
        pthread_mutex_unlock(&chan->lock);
        return NULL; // Maps to Coolscript 'None'
    }

    // Retrieve the pointer and clear the slot
    void* data = chan->buffer[chan->head];
    chan->head = (chan->head + 1) % chan->capacity;
    chan->size--;

    pthread_cond_signal(&chan->not_full);
    pthread_mutex_unlock(&chan->lock);
    
    return data;
}

```

---

## How the Compiler Uses This

When you write Coolscript, the compiler generates calls to these C functions based on its ownership analysis.

### The `send` Operation

In Coolscript: `ch.send(move my_obj)`

1. The compiler's Linear Type Pass verifies `my_obj` isn't used again.
2. The compiler emits a call to `cs_chan_send`.
3. The **Ownership** is now inside the `chan->buffer`. The sender task is effectively "blind" to that memory now.

### The `receive` Operation

In Coolscript: `if let obj = ch.receive():`

1. The compiler emits a call to `cs_chan_receive`.
2. The runtime returns the pointer.
3. The compiler's Type System "wraps" this raw pointer back into the `User` or `List` type.
4. Because the runtime returned it, the receiver task now **owns** that memory and is responsible for eventually freeing it (or moving it again).

---

## Handling the "Clone Sender" Pattern

To support Go's pattern of multiple producers and one consumer, the runtime needs to track how many "Senders" exist so it knows when to actually close the channel.

### Refined Logic:

* **`ch.clone_sender()`**: Increments a reference count on the channel object.
* **`move ch`**: When a channel variable is burned (e.g., at the end of a task), it decrements the count.
* **Closing**: When the reference count reaches zero, the runtime signals `closed = 1`, and all pending `receive` calls return `NULL` (Coolscript `None`).

---

## Performance Considerations

1. **Zero-Copy**: Only the pointer (8 bytes) is ever moved. Even if you "send" a 10GB `List[u8]`, the performance cost is negligible.
2. **No GC Pressure**: Since memory is freed by the owner and not the channel, the channel doesn't have to "scan" its contents.
3. **Static Dispatch**: The `element_size` is known at compile-time, so the compiler can generate specialized versions of these functions for primitives to avoid even the `void*` overhead if needed.
