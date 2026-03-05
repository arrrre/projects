#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "base.h"
#include "arena.h"

// ./main compress test_in.txt test_out.txt

typedef struct {
    u32 n;
    u8 value;
} huffman_occurence;

typedef struct huffman_node {
    u32 code;
    u8 value;
    struct huffman_node* left;
    struct huffman_node* right;
} huffman_node;

u8* string_read(mem_arena* arena, const char* filename);
void string_write(const char* filename, u8* s);

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

        u8* s = string_read(perm_arena, filename_in);

        u32* occurences = PUSH_ARRAY(perm_arena, u32, 256);

        for (u32 i = 0; i < strlen(s); i++) {
            occurences[s[i] - 'a']++;
        }

        for (u32 i = 0; i < 256; i++) {
            printf("%d ", occurences[i]);
        }
        printf("\n");

        u8* cs = NULL;
        string_write(filename_out, cs);
        printf("Compression finished\n");
    } else if (strcmp(mode, "decompress") == 0) {
        printf("decompress\n");
    } else {
        printf("Unknown mode: %s\n", mode);
    }

    arena_destroy(perm_arena);

    return 0;
}

u8* string_read(mem_arena* arena, const char* filename) {
    FILE* f = fopen(filename, "r");

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    u8* s = PUSH_ARRAY(arena, u8, size);
    fread(s, 1, size, f);

    fclose(f);

    return s;
}

void string_write(const char* filename, u8* s) {
    FILE* f = fopen(filename, "w");

    fprintf(f, (const char*)s);

    fclose(f);
}
