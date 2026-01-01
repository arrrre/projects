#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef enum {
    NODE_ADD,
    NODE_SUB,
    NODE_MUL,
    NODE_DIV,
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

const char* supportedFuncs[] = { "sin", "cos", "tan", "exp", "log", "sqrt", "abs" };
const int numFuncs = sizeof(supportedFuncs) / sizeof(char*);

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

    if (isalpha(**s)) {
        char buffer[16];
        int len = 0;
        while (isalpha(**s) && len < 15) {
            buffer[len++] = *(*s)++;
        }
        buffer[len] = '\0';

        if (strcmp(buffer, "x") == 0) {
            return createNode(NODE_VAR);
        }

        for (int i = 0; i < numFuncs; i++) {
            if (strcmp(buffer, supportedFuncs[i]) == 0) {
                Node* node = createNode(NODE_FUNC);
                strcpy(node->funcName, buffer);
                node->left = parseFactor(s);
                return node;
            }
        }

        printf("Error: Unknown function '%s'\n", buffer);
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

Node* parseTerm(const char** s) {
    Node* left = parsePower(s);
    skipSpaces(s);
    while (**s == '*' || **s == '/') {
        char op = **s;
        (*s)++;
        Node* node = createNode(op == '*' ? NODE_MUL : NODE_DIV);
        node->left = left;
        node->right = parsePower(s);
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

Node* buildTree(const char** s, int shouldPrint) {
    Node* root = parseExpression(s);
    if (**s != '\0') {
        freeTree(root);
        return NULL;
    }
    if (shouldPrint != 0) printTree(root, 0, "");
    return root;
}

double evaluate(Node *node, const double x) {
    if (node == NULL) return 0;
    if (node->type == NODE_NUM) return node->value;
    if (node->type == NODE_VAR) return x;
    double left = evaluate(node->left, x), right = evaluate(node->right, x);
    if (node->type == NODE_ADD) return left + right;
    if (node->type == NODE_SUB) return left - right;
    if (node->type == NODE_MUL) return left * right;
    if (node->type == NODE_DIV) return left / right;
    if (node->type == NODE_POW) return pow(left, right);
    if (node->type == NODE_FUNC) {
        if (strcmp(node->funcName, "sin") == 0) return sin(left);
        if (strcmp(node->funcName, "cos") == 0) return cos(left);
        if (strcmp(node->funcName, "tan") == 0) return tan(left);
        if (strcmp(node->funcName, "exp") == 0) return exp(left);
        if (strcmp(node->funcName, "log") == 0) return log(left);
        if (strcmp(node->funcName, "sqrt") == 0) return sqrt(left);
        if (strcmp(node->funcName, "abs") == 0) return fabs(left);
    }
    return 0;
}

float* getFunctionValues(Node *node, const float lower, const float upper, const float step) {
    const int n = (upper - lower) / step;
    float* values = (float *) malloc(n * sizeof(float));
    float x = lower;
    for (int i = 0; i < n; i++) {
        values[i] = evaluate(node, x);
        x += step;
    }
    return values;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./main function <lower> <upper> <step>\nExample: ./main '10*sin(5*x)'\nExample: ./main '10*sin(5*x)' -10 10 0.01\n");
        return 0;
    }
    float lower = 0, upper = 1, step = 0.1;
    int shouldPrint = 0;
    if (argc >= 5) {
        lower = atof(argv[2]); upper = atof(argv[3]); step = atof(argv[4]);
    }
    if (argc >= 6) {
        shouldPrint = atoi(argv[5]);
    }
    if (lower > upper) {
        printf("Lower bound can not be greater than upper bound.\n");
        return 1;
    }
    const char** s = (const char **)&argv[1];
    Node* root = buildTree(s, shouldPrint);
    if (root == NULL) {
        printf("Syntax Error at: %s\n", *s);
        return 1;
    }
    float* values = getFunctionValues(root, lower, upper, step);
    for (int i = 0; i < (upper - lower) / step; i++) printf("%.2f ", values[i]);
    printf("\n");
    free(values);
    freeTree(root);
    return 0;
}
