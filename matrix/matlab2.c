#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "mat.h"

int main() {
	char input[256];
	while (1) {
		printf(">> ");
		if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (strlen(input) == 0) continue;
        if (strcmp(input, "q") == 0) break;
		
	}

	return 0;
}
