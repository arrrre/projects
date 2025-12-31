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
    if (isalpha(**s)) {
        if (strncmp(*s, "sin", 3) == 0) {
            *s += 3;
            Node* node = createNode(NODE_FUNC);
            strcpy(node->funcName, "sin");
            node->left = parseFactor(s);
            return node;
        }
        if (strncmp(*s, "cos", 3) == 0) {
            *s += 3;
            Node* node = createNode(NODE_FUNC);
            strcpy(node->funcName, "cos");
            node->left = parseFactor(s);
            return node;
        }
        if (**s == 'x') {
            (*s)++;
            return createNode(NODE_VAR);
        }
    }
    return NULL;
}

Node* parseTerm(const char** s) {
    Node* left = parseFactor(s);
    skipSpaces(s);
    while (**s == '*' || **s == '/') {
        char op = **s;
        (*s)++;
        Node* node = createNode(op == '*' ? NODE_MUL : NODE_DIV);
        node->left = left;
        node->right = parseFactor(s);
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

double evaluate(Node *node, const double x) {
    if (node == NULL) return 0;
    if (node->type == NODE_NUM) return node->value;
    if (node->type == NODE_VAR) return x;
    double left = evaluate(node->left, x), right = evaluate(node->right, x);
    if (node->type == NODE_ADD) return left + right;
    if (node->type == NODE_SUB) return left - right;
    if (node->type == NODE_MUL) return left * right;
    if (node->type == NODE_DIV) return left / right;
    if (node->type == NODE_FUNC) {
        if (strcmp(node->funcName, "sin") == 0) return sin(left);
        if (strcmp(node->funcName, "cos") == 0) return cos(left);
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
    if (argc == 5) {
        lower = atof(argv[2]); upper = atof(argv[3]); step = atof(argv[4]);
    }
    if (lower > upper) {
        printf("Lower bound can not be greater than upper bound.\n");
        return 0;
    }
    const char **s = (const char **)&argv[1];
    Node *root = parseExpression(s);
    float* values = getFunctionValues(root, lower, upper, step);
    for (int i = 0; i < (upper - lower) / step; i++) printf("%f ", values[i]);
    printf("\n");
    free(values);
    freeTree(root);
    return 0;
}
