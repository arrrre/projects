#include <stdio.h>
#include <stdlib.h>

#include "base.h"
#include "arena.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: ./main (de)compress <filename>\n");
        exit(1);
    }

    const u8* filename = (u8*)argv[1];

    return 0;
}
