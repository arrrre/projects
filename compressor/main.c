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
    u16 unique_chars;
} huff_header;
#pragma pack(pop)

typedef struct {
    u8* file_start;
    u8* data_start;
} header_write_buffer;

typedef struct {
    huff_node* root;
    u64 original_size;
    u8* data_start;
} header_read_buffer;

string8* string_read(mem_arena* arena, const char* filename);
void string_write(const char* filename, string8* s);

string8* compress(mem_arena* arena, string8* s);
string8* decompress(mem_arena* arena, string8* s);

header_write_buffer write_header(mem_arena* arena, u64 original_size, u32* counts);
header_read_buffer read_header(mem_arena* arena, string8* s);

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
    u32 counts[ASCII_UNIQUE] = {0};
    for (u64 i = 0; i < s->size; i++) { counts[s->str[i]]++; }

    min_heap* heap = heapify(arena, counts, ASCII_UNIQUE);
    huff_node* root = treeify(arena, heap);

    code_table* table = PUSH_STRUCT(arena, code_table);
    table->codes = PUSH_ARRAY(arena, code, ASCII_UNIQUE);
    memset(table->codes, 0, sizeof(code) * ASCII_UNIQUE);

    char path_buffer[ASCII_UNIQUE];
    generate_codes(arena, table, root, path_buffer, 0);

    header_write_buffer hb = write_header(arena, s->size, counts);

    bit_writer bw = {0};
    bw.data = hb.data_start;
    bw.data[0] = 0;

    for (u64 i = 0; i < s->size; i++) {
        write_bits(&bw, &table->codes[s->str[i]].bits);
    }

    u64 bit_data_len = bw.byte_idx + (bw.bit_idx > 0 ? 1 : 0);
    
    string8* result = PUSH_STRUCT(arena, string8);
    result->str = hb.file_start;
    result->size = (u64)(hb.data_start - hb.file_start) + bit_data_len;

    return result;
}

string8* decompress(mem_arena* arena, string8* s) {
    header_read_buffer hb = read_header(arena, s);

    bit_reader br = {0};
    br.data = hb.data_start;
    string8* result = PUSH_STRUCT(arena, string8);
    result->size = hb.original_size;
    result->str = PUSH_ARRAY(arena, u8, result->size);

    for (u64 i = 0; i < result->size; i++) {
        huff_node* node = hb.root;

        while (node->left != NULL || node->right != NULL) {
            u8 bit = read_bit(&br);
            node = (bit == 0) ? node->left : node->right;
        }

        result->str[i] = node->value;
    }

    return result;
}

header_write_buffer write_header(mem_arena* arena, u64 original_size, u32* counts) {
    u16 unique_count = 0;
    for (int i = 0; i < ASCII_UNIQUE; i++) {
        if (counts[i] > 0) { unique_count++; }
    }

    u64 header_full_size = sizeof(huff_header) + (unique_count * 5);
    u8* buffer = PUSH_ARRAY(arena, u8, header_full_size + original_size);

    huff_header* header = (huff_header*)buffer;
    header->magic = HEADER_MAGIC;
    header->original_size = original_size;
    header->unique_chars = unique_count;

    u8* cursor = buffer + sizeof(huff_header);

    for (int i = 0; i < ASCII_UNIQUE; i++) {
        if (counts[i] > 0) {
            *cursor++ = (u8)i;
            *(u32*)cursor = counts[i];
            cursor += sizeof(u32);
        }
    }

    return (header_write_buffer){ .file_start = buffer, .data_start = cursor };
}

header_read_buffer read_header(mem_arena* arena, string8* s) {
    u8* cursor = s->str;

    huff_header* header = (huff_header*)cursor;
    cursor += sizeof(huff_header);

    u32 counts[ASCII_UNIQUE] = {0};
    for (u16 i = 0; i < header->unique_chars; i++) {
        u8 character = *cursor;
        cursor++;
        u32 count = *(u32*)cursor;
        cursor += sizeof(u32);

        counts[character] = count;
    }

    min_heap* heap = heapify(arena, counts, ASCII_UNIQUE);
    huff_node* root = treeify(arena, heap);

    return (header_read_buffer) {
        .root = root,
        .original_size = header->original_size,
        .data_start = cursor
    };
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
