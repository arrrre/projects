#include <stdio.h>
#include <string.h>

#include "base.h"
#include "arena.h"
#include "minheap.h"
#include "huffnode.h"

// ./main compress test/test.txt test/test_comp.txt
// ./main decompress test/test_comp.txt test/test_decomp.txt

#define ASCII_UNIQUE 256
#define HEADER_MAGIC 0x46465548 // Hex for "HUFF"

typedef struct {
    u8* str;
    u64 size;
} string8;

#define STR8_LIT(s) (string8){ (u8*)(s), sizeof(s) - 1 }
#define STR8_FMT(s) (int)(s).size, (char*)(s).str

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

typedef struct {
    u8* data;
    u64 byte_idx;
    i32 bit_idx;
} bit_reader;

#pragma pack(push, 1)
typedef struct {
    u32 magic;
    u64 original_size;
    u32 counts[ASCII_UNIQUE];
} huff_header;
#pragma pack(pop)

string8* string_read(mem_arena* arena, const char* filename);
void string_write(const char* filename, string8* s);

string8* compress(mem_arena* arena, string8* s);
string8* decompress(mem_arena* arena, string8* s);

void generate_codes(
    mem_arena* arena, code_table* table,
    huff_node* node, char* path, i32 depth
);

void write_bits(bit_writer* bw, string8* bits);
u8 read_bit(bit_reader* br);

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: ./main (de)compress <input_file> <output_file>\n");
        return 1;
    }

    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    const char* mode = argv[1];
    const char* filename_in = argv[2];
    const char* filename_out = argv[3];

    if (strcmp(mode, "compress") == 0) {
        string8* s = string_read(perm_arena, filename_in);
        string8* cs = compress(perm_arena, s);
        string_write(filename_out, cs);

        printf("%lld bytes -> %lld bytes (%.1f%%)\n", s->size, cs->size,
            (1.0f - (f32)cs->size / s->size) * 100.0f);
    } else if (strcmp(mode, "decompress") == 0) {
        string8* cs = string_read(perm_arena, filename_in);
        string8* s = decompress(perm_arena, cs);
        string_write(filename_out, s);
    } else if (strcmp(mode, "test") == 0 ) {
        string8* s1 = string_read(perm_arena, filename_in);
        string8* cs1 = compress(perm_arena, s1);
        string_write(filename_out, cs1);
        string8* cs2 = string_read(perm_arena, filename_out);
        string8* s2 = decompress(perm_arena, cs2);
        string_write(filename_out, s2);

        if (strcmp((char*)s1->str, (char*)s2->str) == 0) {
            printf("Passed\n");
        } else {
            printf("Failed\n");
        }
    } else {
        printf("Unknown mode: %s\n", mode);
    }

    arena_destroy(perm_arena);

    return 0;
}

string8* string_read(mem_arena* arena, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL) { return NULL; }

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    string8* s = PUSH_STRUCT(arena, string8);
    s->size = size;
    s->str = PUSH_ARRAY(arena, u8, size);

    size_t bytes_read = fread(s->str, 1, size, f);
    
    if (bytes_read != size) {
        printf("Warning: Only read %zu of %llu bytes\n", bytes_read, s->size);
        s->size = bytes_read;
    }

    fclose(f);
    return s;
}

void string_write(const char* filename, string8* s) {
    FILE* f = fopen(filename, "wb");
    if (f == NULL) { return; }

    size_t bytes_written = fwrite(s->str, 1, s->size, f);

    if (bytes_written != s->size) {
        printf("Warning: Only wrote %zu of %llu bytes\n", bytes_written, s->size);
    }

    fclose(f);
}

string8* compress(mem_arena* arena, string8* s) {
    u32* occurences = PUSH_ARRAY(arena, u32, ASCII_UNIQUE);

    for (u64 i = 0; i < s->size; i++) { occurences[s->str[i]]++; }

    min_heap* heap = heapify(arena, occurences, ASCII_UNIQUE);
    huff_node* root = treeify(arena, heap);

    code_table* table = PUSH_STRUCT(arena, code_table);
    table->codes = PUSH_ARRAY(arena, code, ASCII_UNIQUE);
    memset(table->codes, 0, sizeof(code) * ASCII_UNIQUE);

    char path_buffer[ASCII_UNIQUE];
    generate_codes(arena, table, root, path_buffer, 0);

    u64 header_size = sizeof(huff_header);
    u64 max_data_size = s->size;
    
    u8* buffer = PUSH_ARRAY(arena, u8, header_size + max_data_size);
    
    huff_header* header = (huff_header*)buffer;
    header->magic = HEADER_MAGIC;
    header->original_size = s->size;
    memcpy(header->counts, occurences, sizeof(u32) * 256);

    bit_writer bw = {0};
    bw.data = buffer + header_size;
    bw.data[0] = 0;
    bw.bit_idx = 0;
    bw.byte_idx = 0;

    for (u64 i = 0; i < s->size; i++) {
        string8* bits = &table->codes[s->str[i]].bits;
        write_bits(&bw, bits);
    }

    u64 compressed_data_len = bw.byte_idx + (bw.bit_idx > 0 ? 1 : 0);
    string8* result = PUSH_STRUCT(arena, string8);
    result->str = buffer;
    result->size = header_size + compressed_data_len;

    return result;
}

string8* decompress(mem_arena* arena, string8* s) {
    huff_header* header = (huff_header*)s->str;
    if (header->magic != HEADER_MAGIC) { return NULL; }

    min_heap* heap = heapify(arena, header->counts, ASCII_UNIQUE);
    huff_node* root = treeify(arena, heap);

    string8* result = PUSH_STRUCT(arena, string8);
    result->size = header->original_size;
    result->str = PUSH_ARRAY(arena, u8, result->size);

    bit_reader br = {0};
    br.data = s->str + sizeof(huff_header);

    for (u64 i = 0; i < result->size; i++) {
        huff_node* node = root;
        
        while (node->left != NULL || node->right != NULL) {
            u8 bit = read_bit(&br);
            node = (bit == 0) ? node->left : node->right;
        }

        result->str[i] = node->value;
    }

    return result;
}

void generate_codes(
    mem_arena* arena, code_table* table,
    huff_node* node, char* path, i32 depth
) {
    if (node == NULL) { return; }

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

void write_bits(bit_writer* bw, string8* bits) {
    for (u64 i = 0; i < bits->size; i++) {
        if (bits->str[i] == '1') {
            bw->data[bw->byte_idx] |= (1 << (7 - bw->bit_idx));
        }
        
        bw->bit_idx++;

        if (bw->bit_idx == 8) {
            bw->bit_idx = 0;
            bw->data[++bw->byte_idx] = 0;
        }
    }
}

u8 read_bit(bit_reader* br) {
    u8 byte = br->data[br->byte_idx];
    u8 bit = (byte >> (7 - br->bit_idx)) & 1;

    br->bit_idx++;
    if (br->bit_idx == 8) {
        br->bit_idx = 0;
        br->byte_idx++;
    }
    return bit;
}
