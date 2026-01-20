#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <errno.h>
#include <time.h>
#include <windows.h>

HANDLE hStdin = INVALID_HANDLE_VALUE;
DWORD dwOriginalMode = 0;

typedef struct erow {
    int size;          // The number of characters in the row
    int rsize;         // The amount of memory allocated for 'chars'
    char *chars;       // The actual character data
} erow;

struct editorConfig {
    int cx, cy;
    int numlines;
    erow *rows;
    int modified;
    char *filename;
    int screenrows, screencols;
    char statusmsg[80];
    time_t statusmsg_time;
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

void editorClearScreenOnExit() {
    printf("\x1b[2J\x1b[H");
    fflush(stdout); 
}

void updateWindowSize(struct editorConfig *ec) {
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    ec->screenrows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    ec->screencols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
}

enum editorKey {
    ARROW_LEFT = 1000, // Start high to avoid collision with ASCII
    ARROW_RIGHT,
    ARROW_UP,
    ARROW_DOWN,
    HOME,
    END,
    DEL,
    ESC,
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
            case 27: return ESC;
        }
    }
    return c;
}

void editorSetStatusMessage(struct editorConfig *ec, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ec->statusmsg, sizeof(ec->statusmsg), fmt, ap);
    va_end(ap);
    ec->statusmsg_time = time(NULL);
}

void editorRefreshScreen(struct editorConfig* ec) {
    updateWindowSize(ec);
    printf("\x1b[?25l"); // Hide cursor during redraw to prevent flickering
    printf("\x1b[H");    // Move to 1,1

    // We leave 2 rows at the bottom for status and message
    for (int y = 0; y < ec->screenrows - 2; y++) {
        if (y < ec->numlines) {
            // Draw actual file content
            int len = ec->rows[y].size;
            if (len > ec->screencols) len = ec->screencols;
            printf("%.*s", len, ec->rows[y].chars);
        } else {
            // Draw filler for empty space
            printf("~");
        }
        printf("\x1b[K\r\n"); // Clear to end of line and move to next
    }

    // --- Draw Status Bar ---
    printf("\x1b[7m");
    char status[120], rstatus[120];
    int len = snprintf(status, sizeof(status), " %.20s - %d lines %s",
                       ec->filename, ec->numlines, ec->modified ? "(modified)" : "");
    int rlen = snprintf(rstatus, sizeof(rstatus), "Row %d, Col %d ", ec->cy + 1, ec->cx + 1);
    
    if (len > ec->screencols) len = ec->screencols;
    printf("%.*s", len, status);
    while (len < ec->screencols) {
        if (ec->screencols - len == rlen) {
            printf("%s", rstatus);
            break;
        } else {
            printf(" ");
            len++;
        }
    }
    printf("\x1b[m\r\n");

    // --- Draw Message Bar ---
    if (time(NULL) - ec->statusmsg_time < 5) {
        printf("%.*s", ec->screencols, ec->statusmsg);
    }
    printf("\x1b[K"); 

    // Move cursor and show it again
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD final_pos = {(SHORT)ec->cx, (SHORT)ec->cy};
    SetConsoleCursorPosition(hConsole, final_pos);
    printf("\x1b[?25h"); 
}

void editorMoveCursor(struct editorConfig *ec, int key) {
    // Get the current row if it exists
    erow *row = (ec->cy >= ec->numlines) ? NULL : &ec->rows[ec->cy];

    switch (key) {
        case ARROW_LEFT:
            if (ec->cx > 0) {
                ec->cx--;
            } else if (ec->cy > 0) {
                ec->cy--;
                ec->cx = ec->rows[ec->cy].size; // Move to end of previous line
            }
            break;
        case ARROW_RIGHT:
            if (row && ec->cx < row->size) {
                ec->cx++;
            } else if (row && ec->cx == row->size && ec->cy < ec->numlines - 1) {
                ec->cy++;
                ec->cx = 0; // Wrap to start of next line
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
            if (row) ec->cx = row->size;
            break;
    }

    // Boundary check: Ensure cursor isn't past the end of the line after moving Up/Down
    row = (ec->cy >= ec->numlines) ? NULL : &ec->rows[ec->cy];
    int rowlen = row ? row->size : 0;
    if (ec->cx > rowlen) ec->cx = rowlen;
}

void editorInsertRow(struct editorConfig *ec, int at, char *s, size_t len) {
    if (at < 0 || at > ec->numlines) return;

    ec->rows = realloc(ec->rows, sizeof(erow) * (ec->numlines + 1));
    
    // Shift rows down to make space at 'at'
    if (at < ec->numlines) {
        memmove(&ec->rows[at + 1], &ec->rows[at], sizeof(erow) * (ec->numlines - at));
    }

    ec->rows[at].size = (int)len;
    ec->rows[at].rsize = (int)len + 16;
    ec->rows[at].chars = malloc(ec->rows[at].rsize);
    memcpy(ec->rows[at].chars, s, len);
    ec->rows[at].chars[len] = '\0';
    
    ec->numlines++;
}

void editorOpen(struct editorConfig *ec) {
    FILE *fp = fopen(ec->filename, "r");
    if (!fp) {
        editorInsertRow(ec, ec->numlines, "", 0);
        return;
    }

    char buffer[1024];
    int fileEndsWithNewline = 0;
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        size_t len = strlen(buffer);
        // Strip trailing newlines
        while (len > 0 && (buffer[len - 1] == '\n' || buffer[len - 1] == '\r')) {
            fileEndsWithNewline = buffer[len-1] == '\n' ? 1 : 0;
            len--;
        }
        editorInsertRow(ec, ec->numlines, buffer, len);
    }

    fclose(fp);
    
    // If file is empty, ensure there is at least one blank row
    if (ec->numlines == 0 || fileEndsWithNewline) {
        editorInsertRow(ec, ec->numlines, "", 0);
    }
}

char *editorRowsToString(struct editorConfig* ec, int *buflen) {
    int total_len = 0;
    for (int i = 0; i < ec->numlines; i++) {
        total_len += ec->rows[i].size;
        if (i < ec->numlines - 1) total_len++;
    }
    
    if (total_len == 0) {
        *buflen = 0;
        return NULL;
    }

    char *buf = malloc(total_len);
    char *p = buf;

    for (int i = 0; i < ec->numlines; i++) {
        size_t len = ec->rows[i].size;
        memcpy(p, ec->rows[i].chars, len);
        p += len;
        if (i < ec->numlines - 1) {
            *p = '\n';
            p++;
        }
    }

    *buflen = total_len;
    return buf;
}

void editorSave(struct editorConfig* ec) {
    if (ec->numlines == 0) return; // Nothing to save

    int len;
    char *buf = editorRowsToString(ec, &len);
    if (buf == NULL) {
        perror("Memory allocation for save failed");
        return;
    }

    // Use a temporary file handle for writing
    FILE *fp = fopen(ec->filename, "w");
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
        ec->modified = 0;
        printf("[File saved successfully]\n");
    }

    fclose(fp);
    free(buf);
}

void editorRowInsertChar(erow *row, int at, int c) {
    if (at < 0 || at > row->size) at = row->size;
    
    // Only reallocate if we don't have enough capacity
    if (row->size + 1 >= row->rsize) {
        row->rsize = row->size + 8; // Allocate in chunks of 8
        row->chars = realloc(row->chars, row->rsize);
    }
    
    memmove(&row->chars[at + 1], &row->chars[at], row->size - at + 1);
    row->size++;
    row->chars[at] = c;
}

void editorInsertChar(struct editorConfig *ec, int c) {
    if (ec->cy == ec->numlines) {
        editorInsertRow(ec, ec->numlines, "", 0);
    }
    editorRowInsertChar(&ec->rows[ec->cy], ec->cx, c);
    ec->cx++;
    ec->modified++;
}

void editorRowDelChar(erow *row, int at) {
    if (at < 0 || at >= row->size) return;
    
    // Shift characters left to overwrite the char at 'at'
    memmove(&row->chars[at], &row->chars[at + 1], row->size - at);
    row->size--;
}

void editorDelRow(struct editorConfig *ec, int idx) {
    if (idx < 0 || idx >= ec->numlines) return;
    free(ec->rows[idx].chars);
    
    int rows_to_move = ec->numlines - (idx + 1);
    if (rows_to_move > 0) {
        memmove(&ec->rows[idx], &ec->rows[idx + 1], sizeof(erow) * rows_to_move);
    }
    ec->numlines--;
}

void editorBackspaceChar(struct editorConfig *ec) {
    if (ec->cy == ec->numlines) return;
    if (ec->cy == 0 && ec->cx == 0) return;

    erow *row = &ec->rows[ec->cy];
    if (ec->cx > 0) {
        /* Case 1: In the middle or end of a line */
        editorRowDelChar(row, ec->cx - 1);
        ec->cx--;
    } else {
        /* Case 2: At the start of a line - merge with previous line */
        erow *prev_row = &ec->rows[ec->cy - 1];
        
        // 1. Expand the previous row's memory to fit the current row
        int new_size = prev_row->size + row->size;
        prev_row->chars = realloc(prev_row->chars, new_size + 1);
        prev_row->rsize = new_size + 1;
        
        // 2. Copy current row data to the end of previous row
        memcpy(&prev_row->chars[prev_row->size], row->chars, row->size);
        prev_row->size = new_size;
        prev_row->chars[new_size] = '\0'; // Ensure null termination
        
        ec->cx = prev_row->size - row->size;
        editorDelRow(ec, ec->cy); // Delete the current row
        ec->cy--;                 // Move cursor up
    }
    ec->modified++;
}

void editorDeleteChar(struct editorConfig *ec) {
    if (ec->cy == ec->numlines) return;

    erow *row = &ec->rows[ec->cy];
    if (ec->cx < row->size) {
        /* Case 1: Cursor is in the middle of the line */
        editorRowDelChar(row, ec->cx);
    } else {
        /* Case 2: Cursor is at the end of the line - pull NEXT line up */
        
        // If this is the last line, there's no next line to pull up
        if (ec->cy == ec->numlines - 1) return;

        erow *next_row = &ec->rows[ec->cy + 1];

        // 1. Expand current row's memory to fit next_row
        int new_size = row->size + next_row->size;
        row->chars = realloc(row->chars, new_size + 1);
        row->rsize = new_size + 1;

        // 2. Copy next_row data to the end of current row
        memcpy(&row->chars[row->size], next_row->chars, next_row->size);
        row->size = new_size;
        row->chars[new_size] = '\0';

        // 3. Delete the next_row from the array
        editorDelRow(ec, ec->cy + 1);
    }
    ec->modified++;
}

void editorInsertNewline(struct editorConfig *ec) {
    if (ec->cx == 0) {
        /* Case 1: At the start of a line - insert a blank row above */
        editorInsertRow(ec, ec->cy, "", 0);
    } else {
        /* Case 2: In the middle or end of a line - split it */
        erow *row = &ec->rows[ec->cy];
        
        // 1. Create a new row with the characters to the right of the cursor
        editorInsertRow(ec, ec->cy + 1, &row->chars[ec->cx], row->size - ec->cx);
        
        // 2. Re-point 'row' because editorInsertRow might have called realloc on ec->row
        row = &ec->rows[ec->cy];
        
        // 3. Truncate the current row
        row->size = ec->cx;
        row->chars[row->size] = '\0';
        
        // Optional: shrink the memory buffer of the truncated row
        row->rsize = row->size + 1;
        row->chars = realloc(row->chars, row->rsize);
    }

    // Move cursor to the beginning of the new line
    ec->cy++;
    ec->cx = 0;
    ec->modified++;
}

int main(int argc, char *argv[]) {
    struct editorConfig ec = {0};

    char *filename = (argc >= 2) ? argv[1] : "new_file.txt";
    ec.filename = strdup(filename);
    editorOpen(&ec);

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
                editorSave(&ec);
                break;
            case CTRL_Q:
                if (ec.modified > 0) {
                    editorSetStatusMessage(&ec, "Unsaved changes! Save? (y/n/esc)");
                    editorRefreshScreen(&ec); // Force a redraw to show the question
                    while (1) {
                        int c = editorReadKey();
                        if (c == 'y' || c == 'Y') {
                            editorSave(&ec);
                            running = 0;
                            break;
                        } else if (c == 'n' || c == 'N') {
                            running = 0;
                            break;
                        } else if (c == 27) {
                            editorSetStatusMessage(&ec, ""); // Clear the warning
                            break;
                        }
                    }
                } else {
                    running = 0;
                }
                break;
            default:
                if (key_pressed < 1000 && isprint(key_pressed)) {
                    editorInsertChar(&ec, key_pressed);
                }
                break;
        }
    }

    editorClearScreenOnExit();
    disableRawMode();

    for (int i = 0; i < ec.numlines; i++) {
        free(ec.rows[i].chars);
    }
    free(ec.rows);
    free(ec.filename);

    return 0;
}
