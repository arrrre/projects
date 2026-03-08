#ifndef MINHEAP_H
#define MINHEAP_H

#include "base.h"
#include "arena.h"
#include "huffnode.h"

typedef struct {
    huff_node** nodes;
    u64 size;
    u64 capacity;
} min_heap;

min_heap* heapify(mem_arena* arena, u32* occurences, u32 size);
void insert (min_heap* heap, huff_node* node);
huff_node* treeify(mem_arena* arena, min_heap* heap);
huff_node* pop_min(min_heap* heap);

#endif
