#ifndef HUFFNODE_H
#define HUFFNODE_H

#include "base.h"

typedef struct huff_node {
    u8 value;
    u32 count;
    struct huff_node* left;
    struct huff_node* right;
} huff_node;

#endif
