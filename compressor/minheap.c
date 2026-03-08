#include "minheap.h"

void swap(min_heap* heap, u64 i, u64 j) {
    huff_node* temp = heap->nodes[i];
    heap->nodes[i] = heap->nodes[j];
    heap->nodes[j] = temp;
}

void sift_down(min_heap* heap, u64 index) {
    while (1) {
        u64 smallest = index;
        u64 left = 2 * index + 1;
        u64 right = 2 * index + 2;

        if (left < heap->size && heap->nodes[left]->count < heap->nodes[smallest]->count)
            smallest = left;

        if (right < heap->size && heap->nodes[right]->count < heap->nodes[smallest]->count)
            smallest = right;

        if (smallest != index) {
            swap(heap, index, smallest);
            index = smallest;
        } else {
            break;
        }
    }
}

min_heap* heapify(mem_arena* arena, u32* occurences, u32 size) {
    min_heap* heap = PUSH_STRUCT(arena, min_heap);
    heap->capacity = size; 
    heap->nodes = PUSH_ARRAY(arena, huff_node*, heap->capacity);
    heap->size = 0;

    for (u32 i = 0; i < size; i++) {
        if (occurences[i] > 0) {
            huff_node* node = PUSH_STRUCT(arena, huff_node);
            node->value = (u8)i;
            node->count = occurences[i];
            node->left = node->right = NULL;
            heap->nodes[heap->size++] = node;
        }
    }

    if (heap->size > 1) {
        for (i64 i = (heap->size / 2) - 1; i >= 0; i--) {
            sift_down(heap, (u64)i);
        }
    }

    return heap;
}

void insert(min_heap* heap, huff_node* node) {
    u64 index = heap->size++;
    heap->nodes[index] = node;

    while (index > 0) {
        u64 parent = (index - 1) / 2;
        if (heap->nodes[index]->count < heap->nodes[parent]->count) {
            swap(heap, index, parent);
            index = parent;
        } else {
            break;
        }
    }
}

huff_node* treeify(mem_arena* arena, min_heap* heap) {
    while (heap->size > 1) {
        huff_node* a = pop_min(heap);
        huff_node* b = pop_min(heap);

        huff_node* parent = PUSH_STRUCT(arena, huff_node);
        parent->value = 0;
        parent->count = a->count + b->count;
        parent->left = a;
        parent->right = b;

        insert(heap, parent);
    }

    return pop_min(heap);
}

huff_node* pop_min(min_heap* heap) {
    if (heap->size == 0) { return NULL; }

    huff_node* root = heap->nodes[0];
    heap->nodes[0] = heap->nodes[--heap->size];
    sift_down(heap, 0);

    return root;
}
