// Compile with: gcc -Wall -Wextra -Wpedantic -o main main.c

#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void clearScreen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

typedef double (*MathFunc)(double);

typedef struct {
    const char* name;
    MathFunc func;
} FunctionEntry;

FunctionEntry functionTable[] = {
    {"sin", sin},
    {"cos", cos},
    {"tan", tan},
    {"exp", exp},
    {"log", log},
    {"sqrt", sqrt},
    {"abs", fabs}
};

const int numFunctions = sizeof(functionTable) / sizeof(FunctionEntry);

typedef struct {
    const char* name;
    double value;
} ConstantEntry;

ConstantEntry constantTable[] = {
    {"pi",      3.14159265358979323846},
    {"e",       2.71828182845904523536},
    {"deg2rad", 0.01745329251994329577}, // pi / 180
    {"rad2deg", 57.2957795130823208768}, // 180 / pi
};

const int numConstants = sizeof(constantTable) / sizeof(ConstantEntry);

typedef enum {
    NODE_ADD,
    NODE_SUB,
    NODE_MUL,
    NODE_DIV,
    NODE_NEG,
    NODE_POW,
    NODE_NUM,
    NODE_VAR,
    NODE_FUNC
} NodeType;

typedef struct Node {
    NodeType type;
    double value;
    char funcName[10];
    struct Node *left, *right;
} Node;

Node* createNode(NodeType type) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->type = type;
    node->value = 0;
    node->left = node->right = NULL;
    return node;
}

void skipSpaces(const char** s) {
    while (**s == ' ') (*s)++;
}

Node* parseExpression(const char** s);

Node* parseFactor(const char** s) {
    skipSpaces(s);
    if (**s == '(') {
        (*s)++;
        Node* node = parseExpression(s);
        if (**s == ')') (*s)++;
        return node;
    }
    if (isdigit(**s)) {
        Node* node = createNode(NODE_NUM);
        node->value = strtod(*s, (char**)s);
        return node;
    }

    if (isalpha(**s) || **s == '_') {
        char buffer[32];
        int len = 0;
        while ((isalnum(**s) || **s == '_') && len < 31) buffer[len++] = *(*s)++;
        buffer[len] = '\0';

        if (strcmp(buffer, "x") == 0) return createNode(NODE_VAR);

        for (int i = 0; i < numFunctions; i++) {
            if (strcmp(buffer, functionTable[i].name) == 0) {
                Node* node = createNode(NODE_FUNC);
                strcpy(node->funcName, buffer);
                node->left = parseFactor(s);
                return node;
            }
        }

        for (int i = 0; i < numConstants; i++) {
            if (strcmp(buffer, constantTable[i].name) == 0) {
                Node* node = createNode(NODE_NUM);
                node->value = constantTable[i].value;
                return node;
            }
        }

        printf("Error: Unknown function or constant '%s'\n", buffer);
        return NULL;
    }
    return NULL;
}

Node* parsePower(const char** s) {
    Node* left = parseFactor(s);
    skipSpaces(s);
    if (**s == '^') {
        (*s)++;
        Node* node = createNode(NODE_POW);
        node->left = left;
        node->right = parsePower(s); 
        return node;
    }
    return left;
}

Node* parseUnary(const char** s) {
    skipSpaces(s);
    if (**s == '-') {
        (*s)++;
        Node* node = createNode(NODE_NEG);
        node->left = parseUnary(s);
        return node;
    }
    if (**s == '+') {
        (*s)++;
        return parseUnary(s);
    }
    return parsePower(s);
}

Node* parseTerm(const char** s) {
    Node* left = parseUnary(s);
    skipSpaces(s);
    while (**s == '*' || **s == '/') {
        char op = **s;
        (*s)++;
        Node* node = createNode(op == '*' ? NODE_MUL : NODE_DIV);
        node->left = left;
        node->right = parseUnary(s);
        left = node;
        skipSpaces(s);
    }
    return left;
}

Node* parseExpression(const char** s) {
    Node* left = parseTerm(s);
    skipSpaces(s);
    while (**s == '+' || **s == '-') {
        char op = **s;
        (*s)++;
        Node* node = createNode(op == '+' ? NODE_ADD : NODE_SUB);
        node->left = left;
        node->right = parseTerm(s);
        left = node;
        skipSpaces(s);
    }
    return left;
}

void freeTree(Node *node) {
    if (node == NULL) return;
    freeTree(node->left);
    freeTree(node->right);
    free(node);
}

void printTree(Node* node, int level, char* prefix) {
    if (node == NULL) return;

    printf("%s|-- ", prefix);

    switch (node->type) {
        case NODE_ADD:  printf("OP: +\n"); break;
        case NODE_SUB:  printf("OP: -\n"); break;
        case NODE_MUL:  printf("OP: *\n"); break;
        case NODE_DIV:  printf("OP: /\n"); break;
        case NODE_NEG:  printf("UNARY: -\n"); break;
        case NODE_POW:  printf("OP: ^\n"); break;
        case NODE_NUM:  printf("NUMBER: %f\n", node->value); break;
        case NODE_VAR:  printf("VARIABLE: x\n"); break;
        case NODE_FUNC: printf("FUNCTION: %s\n", node->funcName); break;
    }

    char newPrefix[256];
    sprintf(newPrefix, "%s|   ", prefix);
    printTree(node->left, level + 1, newPrefix);
    printTree(node->right, level + 1, newPrefix);
}

Node* buildTree(const char** s) {
    Node* root = parseExpression(s);
    if (**s != '\0') {
        freeTree(root);
        return NULL;
    }
    return root;
}

int treeContainsVariables(Node* node) {
    if (node == NULL) return 0;
    if (node->type == NODE_VAR) return 1;
    return treeContainsVariables(node->left) || treeContainsVariables(node->right);
}

double callMathFunc(Node* node, double value) {
    for (int i = 0; i < numFunctions; i++) {
        if (strcmp(node->funcName, functionTable[i].name) == 0) {
            return functionTable[i].func(value);
        }
    }
    return 0;
}

double evaluate(Node *node, const double x) {
    if (node == NULL) return 0;
    double left = evaluate(node->left, x), right = evaluate(node->right, x);
    switch(node->type) {
        case NODE_NUM: return node->value;
        case NODE_VAR: return x;
        case NODE_ADD: return left + right;
        case NODE_SUB: return left - right;
        case NODE_MUL: return left * right;
        case NODE_DIV: return left / right;
        case NODE_NEG: return -left;
        case NODE_POW: return pow(left, right);
        case NODE_FUNC: return callMathFunc(node, left);
        default: return 0;
    }
}

void printHelp(const float lower, const float upper, const float step) {
    printf("\n--- ARBOR COMMANDS ---\n");
    printf("  [expression]  : Parses and evaluates a function (e.g., sin(x*pi))\n");
    printf("  l=[val]       : Set the lower bound of the range (current: %.2f)\n", lower);
    printf("  u=[val]       : Set the upper bound of the range (current: %.2f)\n", upper);
    printf("  s=[val]       : Set the step size (current: %.2f)\n", step);
    printf("  pt            : Toggle tree visualization\n");
    printf("  pf            : Print availabe functions\n");
    printf("  pc            : Print available constants\n");
    printf("  h             : Show help message\n");
    printf("  c             : Clear screen\n");
    printf("  q             : Quit\n");
    printf("----------------------\n\n");
}

void printFunctions() {
    for (int i = 0; i < numFunctions; i++) {
        printf("%s\n", functionTable[i].name);
    }
}

void printConstants() {
    for (int i = 0; i < numConstants; i++) {
        printf("%s = %f\n", constantTable[i].name, constantTable[i].value);
    }
}

int main() {
    char input[256];
    int shouldPrintTree = 0;
    float lower = 0, upper = 10, step = 1;
    printHelp(lower, upper, step);
    while (1) {
        printf(">> ");
        if (!fgets(input, sizeof(input), stdin)) break;
        input[strcspn(input, "\n")] = 0;
        if (strlen(input) == 0) continue;

        if (strcmp(input, "q") == 0) break;
        if (strcmp(input, "c") == 0) { clearScreen(); continue; }
        if (strcmp(input, "h") == 0) { printHelp(lower, upper, step); continue; }
        char* p;
        if ((p = strstr(input, "l=")) != NULL) {
            lower = atof(p + 2);
            printf("Lower: %.2f\n", lower);
            continue;
        }
        if ((p = strstr(input, "u=")) != NULL) {
            upper = atof(p + 2);
            printf("Upper: %.2f\n", upper);
            continue;
        }
        if ((p = strstr(input, "s=")) != NULL) {
            float newStep = atof(p + 2);
            if (newStep <= 0) {
                printf("Step size must be positive\n");
                continue;
            }
            step = newStep;
            printf("Step: %.2f\n", step);
            continue;
        }
        if (strcmp(input, "pt") == 0) {
            shouldPrintTree = !shouldPrintTree;
            printf("PrintTree: %s\n", shouldPrintTree ? "ON" : "OFF");
            continue;
        }
        if (strcmp(input, "pf") == 0) { printFunctions(); continue; }
        if (strcmp(input, "pc") == 0) { printConstants(); continue; }

        const char* s = input;
        Node* root = buildTree(&s);

        if (root == NULL) {
            printf("Error: Could not build tree. Syntax error at: %s.\n", s);
            continue;
        }

        if (shouldPrintTree) printTree(root, 0, "");

        if (treeContainsVariables(root) == 0) {
            printf("%.4f\n", evaluate(root, 0));
        } else {
            printf("Evaluating from %.2f to %.2f (step %.2f):\n", lower, upper, step);
            int numSteps = (int)((upper - lower) / step) + 1;
            for (int i = 0; i < numSteps; i++) {
                float x = lower + (i * step);
                float result = evaluate(root, x);
                printf("f(%.2f) = %.4f\n", x, result);
            }
        }

        freeTree(root);
    }
    return 0;
}
