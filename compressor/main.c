#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "base.h"
#include "arena.h"

// ./main compress test/test.txt test/test_comp.txt
// ./main decompress test/test_comp.txt test/test_decomp.txt

typedef struct {
    u8* str;
    u64 size;
} string8;

#define STR8_LIT(s) (string8){ (u8*)(s), sizeof(s) - 1 }
#define STR8_FMT(s) (int)(s).size, (char*)(s).str

typedef struct list_node {
    u8 value;
    u32 occs;
    struct list_node* next;
    struct list_node* prev;
    struct list_node* left;
    struct list_node* right;
} list_node;

typedef struct {
    string8 bits;
} code;

typedef struct {
    code* codes;
} code_table;

typedef struct {
    u8* data;
    u64 byte_idx;
    i32 bit_idx;
} bit_writer;

string8* string_read(mem_arena* arena, const char* filename);
void string_write(const char* filename, string8* s);

string8* compress(mem_arena* arena, string8* s);
string8* decompress(mem_arena* arena, string8* s);

void generate_codes(
    mem_arena* arena, code_table* table,
    list_node* node, char* path, i32 depth
);
void write_code(bit_writer* bw, string8* bits);

void print_tree(list_node* ln, u32 level);

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: ./main (de)compress <input_file> <output_file>\n");
        exit(1);
    }

    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    const char* mode = argv[1];
    const char* filename_in = argv[2];
    const char* filename_out = argv[3];

    if (strcmp(mode, "compress") == 0) {
        printf("Compression of %s into %s started\n", filename_in, filename_out);

        string8* s = string_read(perm_arena, filename_in);

        string8* cs = compress(perm_arena, s);

        string_write(filename_out, cs);

        printf("Compression finished\n");
    } else if (strcmp(mode, "decompress") == 0) {
        printf("Decompression of %s into %s started\n", filename_in, filename_out);

        string8* s = string_read(perm_arena, filename_in);

        string8* cs = decompress(perm_arena, s);

        string_write(filename_out, cs);

        printf("Decompression finished\n");
    } else {
        printf("Unknown mode: %s\n", mode);
    }

    arena_destroy(perm_arena);

    return 0;
}

string8* string_read(mem_arena* arena, const char* filename) {
    FILE* f = fopen(filename, "r");

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    string8* s = PUSH_STRUCT(arena, string8);
    s->size = size;
    s->str = PUSH_ARRAY(arena, u8, size);

    fread(s->str, 1, size, f);

    fclose(f);

    return s;
}

void string_write(const char* filename, string8* s) {
    FILE* f = fopen(filename, "w");

    fprintf(f, (char*)s->str);

    fclose(f);
}

string8* compress(mem_arena* arena, string8* s) {
    const u32 ASCII_UNIQUE = 256;
    u32* occurences = PUSH_ARRAY(arena, u32, ASCII_UNIQUE);

    for (u64 i = 0; i < s->size; i++) { occurences[s->str[i]]++; }

    list_node* head = PUSH_STRUCT(arena, list_node);
    head->next = head->prev = NULL;
    list_node* tail = head;

    u32 unique = 0;
    for (u32 i = 0; i < ASCII_UNIQUE; i++) {
        u32 max_i = 0;
        for (u32 j = 0; j < ASCII_UNIQUE; j++) {
            if (occurences[j] > occurences[max_i]) { max_i = j; }
        }

        u32 occs = occurences[max_i];
        if (occs == 0) { break; }

        occurences[max_i] = 0;
        unique++;
        
        list_node* ln = PUSH_STRUCT(arena, list_node);
        ln->value = max_i;
        ln->occs = occs;
        ln->next = NULL;
        ln->prev = (struct list_node*)tail;
        tail->next = (struct list_node*)ln;
        tail = ln;
    }

    u32 nodes_left = unique;
    while (nodes_left > 1) {
        list_node* a = tail;
        list_node* b = tail->prev;

        tail = b->prev;
        if (tail) { tail->next = NULL; }

        list_node* parent = PUSH_STRUCT(arena, list_node);
        parent->occs = a->occs + b->occs;
        parent->left = a;
        parent->right = b;

        if (tail == NULL) {
            tail = parent;
            head = parent;
        } else {
            list_node* scan = head;
            while (scan->next && scan->next->occs > parent->occs) {
                scan = scan->next;
            }

            parent->next = scan->next;
            parent->prev = scan;
            if (scan->next) { scan->next->prev = parent; }
            scan->next = parent;

            if (parent->next == NULL) tail = parent;
        }

        nodes_left--;
    }

    list_node* root = (nodes_left == 1) ? tail : NULL;

    // print_tree(root, 0);

    code_table* table = PUSH_STRUCT(arena, code_table);
    table->codes = PUSH_ARRAY(arena, code, 256);
    memset(table->codes, 0, sizeof(code) * 256);
    
    char path_buffer[256];
    generate_codes(arena, table, root, path_buffer, 0);

    bit_writer bw = {0};
    bw.data = PUSH_ARRAY(arena, u8, s->size);
    bw.data[0] = 0;
    bw.bit_idx = 0;
    bw.byte_idx = 0;

    for (u64 i = 0; i < s->size; i++) {
        u8 c = s->str[i];
        string8 bits = table->codes[c].bits;
        write_code(&bw, &bits);
    }

    u64 final_size_in_bytes = bw.byte_idx + (bw.bit_idx > 0 ? 1 : 0);

    printf("original size: %lld, final size: %lld\n", s->size, final_size_in_bytes);

    string8* cs = s;

    return cs;
}

string8* decompress(mem_arena* arena, string8* s) {
    string8* ds = s;

    return ds;
}

void generate_codes(mem_arena* arena, code_table* table, list_node* node, char* path, i32 depth) {
    if (node == NULL) return;

    if (node->left == NULL && node->right == NULL) {
        table->codes[node->value].bits.size = depth;

        u8* str = PUSH_ARRAY(arena, u8, depth);
        memcpy(str, path, depth);
        table->codes[node->value].bits.str = str;

        return;
    }

    path[depth] = '0';
    generate_codes(arena, table, node->left, path, depth + 1);

    path[depth] = '1';
    generate_codes(arena, table, node->right, path, depth + 1);
}

void write_code(bit_writer* bw, string8* bits) {
    for (int i = 0; bits->size; i++) {
        if (bits->str[i] == '1') {
            bw->data[bw->byte_idx] |= (1 << (7 - bw->bit_idx));
        }
        
        bw->bit_idx++;

        if (bw->bit_idx == 8) {
            bw->bit_idx = 0;
            bw->data[bw->byte_idx++] = 0;
        }
    }
}

void print_tree(list_node* ln, u32 level) {
    if (ln == NULL) { return; }

    print_tree(ln->right, level + 1);

    for (u32 i = 0; i < level; i++) { printf("    "); }

    if (ln->left == NULL && ln->right == NULL) {
        printf("---['%c': %d]\n", ln->value, ln->occs);
    } else {
        printf("---(%d)\n", ln->occs);
    }

    print_tree(ln->left, level + 1);
}
