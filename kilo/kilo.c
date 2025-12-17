#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>
#include <windows.h>
#include <errno.h>

#define KILO_VERSION "0.0.1"
#define ABUF_INIT {NULL, 0}

HANDLE hStdin = INVALID_HANDLE_VALUE;
DWORD dwOriginalMode = 0;
int editor_cx = 0; // Current cursor X (column)
int editor_cy = 0; // Current cursor Y (row)

struct editorConfig {
    int cx, cy;       // Cursor position (col, row)
    int numlines;     // Number of lines in the file
    char **filelines; // Pointer to an array of string pointers (the file content)
};

struct editorConfig E;

void initEditor() {
    E.cx = 0;
    E.cy = 0;
    E.numlines = 0;
    E.filelines = NULL;
}

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
    CTRL_S = 19
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

void editorRefreshScreen() {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    printf("\x1b[2J"); 
    printf("\x1b[H"); 

    for (int i = 0; i < E.numlines; i++) {
        printf("%s", E.filelines[i]);
        printf("\r\n"); 
    }

    COORD final_pos = {E.cx, E.cy};
    SetConsoleCursorPosition(hConsole, final_pos);
}

void editorMoveCursor(int key) {
    switch (key) {
        case ARROW_LEFT:
            if (E.cx > 0) {
                E.cx--;
            } else if (E.cy > 0) {
                E.cy--;
                E.cx = strlen(E.filelines[E.cy]);
            }
            break;
        case ARROW_RIGHT:
            if (E.cx < strlen(E.filelines[E.cy])) {
                E.cx++;
            } else if (E.cy < E.numlines - 1) {
                E.cy++;
                E.cx = 0;
            }
            break;
        case ARROW_UP:
            if (E.cy > 0) E.cy--;
            break;
        case ARROW_DOWN:
            if (E.cy < E.numlines - 1) E.cy++;
            break;
        case HOME:
            E.cx = 0;
            break;
        case END:
            E.cx = strlen(E.filelines[E.cy]);
            break;
    }
    
    if (E.cy < E.numlines) {
        size_t linelen = strlen(E.filelines[E.cy]);
        if (E.cx > linelen) E.cx = linelen;
    }
}

void editorMoveCursorBack() {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD pos = {strlen(E.filelines[E.numlines - 1]), E.numlines - 1};
    SetConsoleCursorPosition(hConsole, pos);
}

void editorAppendRow(char *s) {
    E.filelines = realloc(E.filelines, sizeof(char *) * (E.numlines + 1));
    E.filelines[E.numlines] = strdup(s);
    E.numlines++;
}

void editorOpen(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("fopen");
        return;
    }

    char *line = NULL;
    size_t linecap = 0;
    ssize_t linelen;

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        // Remove the trailing newline character, if present
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
        editorAppendRow(buffer);
    }

    fclose(fp);
}

char *editorRowsToString(int *buflen) {
    // 1. Calculate the total size needed for the entire file content
    int total_len = 0;
    for (int i = 0; i < E.numlines; i++) {
        // Line length + 1 for the newline character (\n)
        total_len += strlen(E.filelines[i]) + 1; 
    }
    *buflen = total_len;

    // 2. Allocate one large buffer to hold the entire file
    char *buf = malloc(total_len);
    if (buf == NULL) return NULL; // Handle allocation failure

    char *p = buf; // Pointer to track current position in the buffer
    
    // 3. Copy each line into the buffer, adding a newline
    for (int i = 0; i < E.numlines; i++) {
        size_t len = strlen(E.filelines[i]);
        
        // Copy the line content
        memcpy(p, E.filelines[i], len);
        p += len;
        
        // Add the newline character
        *p = '\n';
        p++;
    }

    return buf;
}

void editorSave(const char *filename) {
    if (E.numlines == 0) return; // Nothing to save

    int len;
    char *buf = editorRowsToString(&len);
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

void editorInsertChar(int c) {
    if (E.cy >= E.numlines) {
        editorAppendRow(""); 
    }

    char *line = E.filelines[E.cy];
    size_t len = strlen(line);
    E.filelines[E.cy] = realloc(line, len + 2); 
    line = E.filelines[E.cy]; // Update the line pointer after realloc
    memmove(&line[E.cx + 1], &line[E.cx], len - E.cx + 1);
    line[E.cx] = (char)c;
    E.cx++;
}

void editorDelChar() {
    if (E.cy == 0 && E.cx == 0) return;
    if (E.cy >= E.numlines) return;
    if (E.cx > 0) {
        char *line = E.filelines[E.cy];
        size_t len = strlen(line);
        memmove(&line[E.cx - 1], &line[E.cx], len - E.cx + 1);
        E.filelines[E.cy] = realloc(line, len);
        E.cx--;
    } else {
        // Case 2: Cursor is at the start of a line (E.cx == 0) -> Join with the previous line
        
        // This is complex and involves:
        // a) Getting the previous line's length.
        // b) Appending the current line's text to the previous line.
        // c) Deleting the current line from the filelines array.
        // d) Moving the cursor to the point of the join.
        
        // **For simplicity, let's skip line joining for the initial version.**
        // Just prevent backspace if E.cx == 0 for now.
    }
}

void editorInsertNewline() {
    if (E.cy == E.numlines - 1 && E.cx == strlen(E.filelines[E.cy])) {
        editorAppendRow("");
    } else {
        // Case 1: Split the current line
        char *current_line = E.filelines[E.cy];
        size_t current_len = strlen(current_line);
        
        // --- A. Create the new line (text *after* the cursor) ---
        // Duplicate the portion of the string starting at E.cx
        // strdup will allocate memory for the new line.
        char *new_line_text = strdup(&current_line[E.cx]);

        // --- B. Truncate the current line (text *before* the cursor) ---
        // The current line ends where the new line begins, plus the null terminator.
        // Reallocate to save memory and ensure correct string termination.
        E.filelines[E.cy] = realloc(current_line, E.cx + 1);
        E.filelines[E.cy][E.cx] = '\0'; // Manually null-terminate the truncated line

        // --- C. Make room in the filelines array for the new line ---
        // Resize the array of char* pointers to hold one more line.
        E.filelines = realloc(E.filelines, sizeof(char *) * (E.numlines + 1));

        // Shift all line pointers *after* the current line one position down
        // The shift starts at E.cy + 1 and moves E.numlines - E.cy pointers.
        memmove(&E.filelines[E.cy + 2], &E.filelines[E.cy + 1], 
                sizeof(char *) * (E.numlines - E.cy));

        // --- D. Insert the new line pointer ---
        E.filelines[E.cy + 1] = new_line_text;
        E.numlines++;
    }

    // --- E. Update Cursor Position ---
    E.cy++;  // Move down one row to the new line
    E.cx = 0; // Move cursor to the beginning of the new line
}

int main(int argc, char *argv[]) {
    initEditor();
    const char *filename = (argc >= 2) ? argv[1] : "new_file.txt"; 
    if (argc >= 2) editorOpen(argv[1]);

    enableRawMode();

    int key_pressed;
    do {
        switch (key_pressed) {
            case ARROW_UP:
            case ARROW_DOWN:
            case ARROW_LEFT:
            case ARROW_RIGHT:
            case HOME:
            case END:
                editorMoveCursor(key_pressed);
                break;
            case BACKSPACE:
                editorDelChar();
                break;
            case ENTER:
                editorInsertNewline();
                break;
            case CTRL_S:
                editorSave(filename);
                break;
            default:
                if (key_pressed != 0) editorInsertChar(key_pressed);
                break;
        }
        editorRefreshScreen();
    } while ((key_pressed = editorReadKey()) != 'q');

    editorMoveCursorBack();
    disableRawMode();

    // Clean up memory before exit
    for (int i = 0; i < E.numlines; i++) {
        free(E.filelines[i]);
    }
    free(E.filelines);

    return 0;
}