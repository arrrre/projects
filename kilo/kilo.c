#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <windows.h>
#include <errno.h>

HANDLE hStdin = INVALID_HANDLE_VALUE;
DWORD dwOriginalMode = 0;

struct editorConfig {
    int cx, cy;       // Cursor position (col, row)
    int numlines;     // Number of lines in the file
    char **filelines; // Pointer to an array of string pointers (the file content)
};

void enableRawMode() {
    hStdin = GetStdHandle(STD_INPUT_HANDLE);
    if (hStdin == INVALID_HANDLE_VALUE) {
        perror("GetStdHandle failed");
        return;
    }
    if (!GetConsoleMode(hStdin, &dwOriginalMode)) {
        perror("GetConsoleMode failed");
        return;
    }
    DWORD dwNewMode = dwOriginalMode;
    dwNewMode &= ~ENABLE_LINE_INPUT;
    dwNewMode &= ~ENABLE_ECHO_INPUT;
    // dwNewMode &= ~ENABLE_PROCESSED_INPUT; // Disable Ctrl+C handling, etc.
    // dwNewMode &= ~ENABLE_WINDOW_INPUT;    // Disable resizing events
    if (!SetConsoleMode(hStdin, dwNewMode)) {
        perror("SetConsoleMode failed");
        return;
    }
}

void disableRawMode() {
    if (hStdin != INVALID_HANDLE_VALUE) {
        SetConsoleMode(hStdin, dwOriginalMode);
    }
}

void editorMoveCursorBack(struct editorConfig* ec) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD pos = {strlen(ec->filelines[ec->numlines - 1]), ec->numlines - 1};
    SetConsoleCursorPosition(hConsole, pos);
}

enum editorKey {
    ARROW_LEFT = 1000, // Start high to avoid collision with ASCII
    ARROW_RIGHT,
    ARROW_UP,
    ARROW_DOWN,
    HOME,
    END,
    DEL,
    BACKSPACE = 8,
    ENTER = 13,
    CTRL_S = 19,
    CTRL_Q = 17
};

int editorReadKey() {
    int c = _getch();
    if (c == 0 || c == 0xE0) {
        c = _getch(); 
        // Map the Windows scan codes to our custom enumeration
        switch (c) {
            case 75: return ARROW_LEFT;
            case 77: return ARROW_RIGHT;
            case 72: return ARROW_UP;
            case 80: return ARROW_DOWN;
            case 71: return HOME;
            case 79: return END;
            case 83: return DEL;
        }
    }
    return c;
}

void editorRefreshScreen(struct editorConfig* ec) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    
    // \x1b[2J clears the screen, \x1b[3J clears the scrollback buffer
    // \x1b[H moves cursor to 1,1
    printf("\x1b[2J\x1b[3J\x1b[H");

    for (int i = 0; i < ec->numlines; i++) {
        printf("%s\r\n", ec->filelines[i]);
    }

    COORD final_pos = {(SHORT)ec->cx, (SHORT)ec->cy};
    SetConsoleCursorPosition(hConsole, final_pos);
}

void editorMoveCursor(struct editorConfig* ec, int key) {
    switch (key) {
        case ARROW_LEFT:
            if (ec->cx > 0) {
                ec->cx--;
            } else if (ec->cy > 0) {
                ec->cy--;
                ec->cx = strlen(ec->filelines[ec->cy]);
            }
            break;
        case ARROW_RIGHT:
            if (ec->cx < strlen(ec->filelines[ec->cy])) {
                ec->cx++;
            } else if (ec->cy < ec->numlines - 1) {
                ec->cy++;
                ec->cx = 0;
            }
            break;
        case ARROW_UP:
            if (ec->cy > 0) ec->cy--;
            break;
        case ARROW_DOWN:
            if (ec->cy < ec->numlines - 1) ec->cy++;
            break;
        case HOME:
            ec->cx = 0;
            break;
        case END:
            ec->cx = strlen(ec->filelines[ec->cy]);
            break;
    }
    
    if (ec->cy < ec->numlines) {
        size_t linelen = strlen(ec->filelines[ec->cy]);
        if (ec->cx > linelen) ec->cx = linelen;
    }
}

void editorAppendRow(struct editorConfig* ec, char *s) {
    ec->filelines = realloc(ec->filelines, sizeof(char *) * (ec->numlines + 1));
    ec->filelines[ec->numlines] = strdup(s);
    ec->numlines++;
}

void editorOpen(struct editorConfig* ec, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return;

    char buffer[1024];
    int fileEndsWithNewline = 0;
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        size_t len = strlen(buffer);
        while (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
            fileEndsWithNewline = buffer[len-1] == '\n' ? 1 : 0;
            len--;
            buffer[len] = '\0';
        }
        editorAppendRow(ec, buffer);
    }
    
    if (fileEndsWithNewline) {
        editorAppendRow(ec, "");
    }

    fclose(fp);
}

char *editorRowsToString(struct editorConfig* ec, int *buflen) {
    int total_len = 0;
    for (int i = 0; i < ec->numlines; i++) {
        total_len += strlen(ec->filelines[i]);
        if (i < ec->numlines - 1) total_len++;
    }
    
    if (total_len == 0) {
        *buflen = 0;
        return NULL;
    }

    char *buf = malloc(total_len);
    char *p = buf;

    for (int i = 0; i < ec->numlines; i++) {
        size_t len = strlen(ec->filelines[i]);
        memcpy(p, ec->filelines[i], len);
        p += len;
        if (i < ec->numlines - 1) {
            *p = '\n';
            p++;
        }
    }

    *buflen = total_len;
    return buf;
}

void editorSave(struct editorConfig* ec, const char *filename) {
    if (ec->numlines == 0) return; // Nothing to save

    int len;
    char *buf = editorRowsToString(ec, &len);
    if (buf == NULL) {
        perror("Memory allocation for save failed");
        return;
    }

    // Use a temporary file handle for writing
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Failed to open file for writing");
        free(buf);
        return;
    }

    // Write the entire buffer content to the file
    size_t bytes_written = fwrite(buf, 1, len, fp);

    if (ferror(fp)) {
        perror("Error writing to file");
    } else if (bytes_written != len) {
        fprintf(stderr, "Warning: Only wrote %zu of %d bytes.\n", bytes_written, len);
    } else {
        printf("[File saved successfully]\n");
    }

    fclose(fp);
    free(buf);
}

void editorInsertChar(struct editorConfig* ec, int c) {
    if (ec->cy >= ec->numlines) {
        editorAppendRow(ec, ""); 
    }

    char *line = ec->filelines[ec->cy];
    size_t len = strlen(line);
    ec->filelines[ec->cy] = realloc(line, len + 2); 
    line = ec->filelines[ec->cy]; // Update the line pointer after realloc
    memmove(&line[ec->cx + 1], &line[ec->cx], len - ec->cx + 1);
    line[ec->cx] = (char)c;
    ec->cx++;
}

void editorBackspaceChar(struct editorConfig* ec) {
    if (ec->cy == 0 && ec->cx == 0) return;
    if (ec->cy >= ec->numlines) return;
    if (ec->cx > 0) {
        char *line = ec->filelines[ec->cy];
        size_t len = strlen(line);
        memmove(&line[ec->cx - 1], &line[ec->cx], len - ec->cx + 1);
        ec->filelines[ec->cy] = realloc(line, len);
        ec->cx--;
    } else {
        // Case 2: Cursor is at the start of a line (ec->cx == 0) -> Join with the previous line
        
        // 1. Get pointers to the previous line and the current line
        char *prev_line = ec->filelines[ec->cy - 1];
        char *curr_line = ec->filelines[ec->cy];
        size_t prev_len = strlen(prev_line);
        size_t curr_len = strlen(curr_line);

        // 2. Reallocate the previous line to hold both lines + null terminator
        prev_line = realloc(prev_line, prev_len + curr_len + 1);
        ec->filelines[ec->cy - 1] = prev_line;

        // 3. Copy current line content to the end of the previous line
        memcpy(&prev_line[prev_len], curr_line, curr_len + 1);

        // 4. Update cursor position: it goes to the end of the "old" previous line
        ec->cx = (int)prev_len;
        ec->cy--;

        // 5. Delete the current line pointer from the array
        // First, free the memory of the line we just merged
        free(curr_line);
        
        // Shift all subsequent line pointers up by one
        int lines_to_move = ec->numlines - (ec->cy + 2);
        if (lines_to_move > 0) {
            memmove(&ec->filelines[ec->cy + 1], &ec->filelines[ec->cy + 2], 
                    sizeof(char *) * lines_to_move);
        }
        
        ec->numlines--;
    }
}

void editorDeleteChar(struct editorConfig* ec) {
    printf("DELETE\n");
    exit(0);
}

void editorInsertNewline(struct editorConfig* ec) {
    if (ec->cy == ec->numlines - 1 && ec->cx == strlen(ec->filelines[ec->cy])) {
        editorAppendRow(ec, "");
    } else {
        // Case 1: Split the current line
        char *current_line = ec->filelines[ec->cy];
        size_t current_len = strlen(current_line);
        
        // Duplicate the portion of the string starting at ec->cx
        // strdup will allocate memory for the new line
        char *new_line_text = strdup(&current_line[ec->cx]);

        // The current line ends where the new line begins, plus the null terminator.
        // Reallocate to save memory and ensure correct string termination.
        ec->filelines[ec->cy] = realloc(current_line, ec->cx + 1);
        ec->filelines[ec->cy][ec->cx] = '\0'; // Manually null-terminate the truncated line

        // Resize array
        ec->filelines = realloc(ec->filelines, sizeof(char *) * (ec->numlines + 1));
        
        // Shift lines down by 1 to make room at cy + 1
        // We move (TotalLines - (CurrentLine + 1)) elements
        int num_elements_to_move = ec->numlines - (ec->cy + 1);
        if (num_elements_to_move > 0) {
            memmove(&ec->filelines[ec->cy + 2], &ec->filelines[ec->cy + 1], 
                    sizeof(char *) * num_elements_to_move);
        }
            
        ec->filelines[ec->cy + 1] = new_line_text;
        ec->numlines++;
    }

    // --- Update Cursor Position ---
    ec->cy++;  // Move down one row to the new line
    ec->cx = 0; // Move cursor to the beginning of the new line
}

int main(int argc, char *argv[]) {
    struct editorConfig ec = {0, 0, 0, NULL};

    const char *filename = (argc >= 2) ? argv[1] : "new_file.txt";
    if (argc >= 2) editorOpen(&ec, filename);

    enableRawMode();

    int running = 1;
    while (running) {
        editorRefreshScreen(&ec);

        int key_pressed = editorReadKey();
        switch (key_pressed) {
            case ARROW_UP:
            case ARROW_DOWN:
            case ARROW_LEFT:
            case ARROW_RIGHT:
            case HOME:
            case END:
                editorMoveCursor(&ec, key_pressed);
                break;
            case BACKSPACE:
                editorBackspaceChar(&ec);
                break;
            case DEL:
                editorDeleteChar(&ec);
                break;
            case ENTER:
                editorInsertNewline(&ec);
                break;
            case CTRL_S:
                editorSave(&ec, filename);
                break;
            case CTRL_Q:
                running = 0;
                break;
            default:
                if (key_pressed < 1000 && isprint(key_pressed)) {
                    editorInsertChar(&ec, key_pressed);
                }
                break;
        }
    }

    editorMoveCursorBack(&ec);
    disableRawMode();

    for (int i = 0; i < ec.numlines; i++) {
        free(ec.filelines[i]);
    }
    free(ec.filelines);

    return 0;
}
