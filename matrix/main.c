#include <stdio.h>

#include "mat.h"

typedef enum {
	MV_FLAG_NONE = 0,

	MV_FLAG_REQUIRES_GRAD  = (1 << 0),
	MV_FLAG_PARAMETER      = (1 << 1),
	MV_FLAG_INPUT          = (1 << 2),
	MV_FLAG_OUTPUT         = (1 << 3),
	MV_FLAG_DESIRED_OUTPUT = (1 << 4),
	MV_FLAG_COST           = (1 << 5),
} model_var_flags;

typedef enum {
	MV_OP_NULL = 0,
	MV_OP_CREATE,

	_MV_OP_UNARY_START,

	MV_OP_RELU,
	MV_OP_SOFTMAX,

	_MV_OP_BINARY_START,

	MV_OP_ADD,
	MV_OP_SUB,
	MV_OP_MATMUL,
	MV_OP_CROSS_ENTROPY,
} model_var_op;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

typedef struct {
	int index;
	int flags;

	matrix_t* val;
	matrix_t* grad;

	model_var_op op;
	struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct {
	model_var** vars;
	int size;
} model_program;

typedef struct {
	int num_vars;

	model_var* input;
	model_var* output;
	model_var* desired_output;
	model_var* cost;

	model_program forward_prog;
	model_program cost_prog;
} model_context;

typedef struct {
	matrix_t* train_images;
	matrix_t* train_labels;
	matrix_t* test_images;
	matrix_t* test_labels;

	int epochs;
	int batch_size;
	float learning_rate;
} model_training_desc;

model_var* mv_create(model_context* model, int rows, int cols, int flags);
model_var* mv_relu(model_context* model, model_var* input, int flags);
model_var* mv_softmax(model_context* model, model_var* input, int flags);
model_var* mv_add(model_context* model, model_var* a, model_var* b, int flags);
model_var* mv_sub(model_context* model, model_var* a, model_var* b, int flags);
model_var* mv_matmul(model_context* model, model_var* a, model_var* b, int flags);
model_var* mv_cross_entropy(model_context* model, model_var* p, model_var* q, int flags);

model_program model_prog_create(model_context* model, model_var* out_var);
void model_prog_compute(model_program* prog);
void model_prog_compute_grads(model_program* prog);

model_context* model_create();
void model_compile(model_context* model);
void model_feedforward(model_context* model);
void model_train(
	model_context* model,
	const model_training_desc* training_desc
);

void draw_mnist_digit(float* data, int image_rows, int image_cols);

int main() {
	const int image_rows = 28, image_cols = 28;
	const int image_size = image_rows * image_cols;
	const int train_set_size = 60000;
	const int test_set_size = 10000;
	const int num_classes = 10;
	matrix_t* train_images = mat_load(train_set_size, image_size, "data/train_images.mat");
	matrix_t* test_images = mat_load(test_set_size, image_size, "data/test_images.mat");
	matrix_t* train_labels = mat_create(train_set_size, num_classes);
	matrix_t* test_labels = mat_create(test_set_size, num_classes);
	
	{
		matrix_t* train_labels_file = mat_load(train_set_size, 1, "data/train_labels.mat");
		matrix_t* test_labels_file = mat_load(test_set_size, 1, "data/test_labels.mat");

		for (int i = 0; i < train_set_size; i++) {
			unsigned int num = train_labels_file->data[i];
			train_labels->data[i * num_classes + num] = 1.0f;
		}

		for (int i = 0; i < test_set_size; i++) {
			unsigned int num = test_labels_file->data[i];
			test_labels->data[i * num_classes + num] = 1.0f;
		}

		mat_free(train_labels_file);
		mat_free(test_labels_file);
	}

	draw_mnist_digit(train_images->data, image_rows, image_cols);
	for (int i = 0; i < num_classes; i++) {
		printf("%.0f ", train_labels->data[i]);
	}
	printf("\n");

	mat_free(train_images);
	mat_free(test_images);
	mat_free(train_labels);
	mat_free(test_labels);

	return 0;
}

void draw_mnist_digit(float* data, int image_rows, int image_cols) {
	for (int y = 0; y < image_rows; y++) {
		for (int x = 0; x < image_cols; x++) {
			float num = data[x + y * image_cols];
			// Some grayscale color thing... 0-255
			unsigned int col = 232 + (unsigned int)(num * 24);
			printf("\x1b[48;5;%dm  ", col);
		}
		printf("\n");
	}
	printf("\x1b[0m");
}

model_var* mv_create(
	model_context* model, int rows, int cols, int flags
) {
	model_var* out = malloc(sizeof(*out));

	if (out == NULL) return NULL;

	out->index = model->num_vars++;
	out->flags = flags;
	out->op = MV_OP_CREATE;
	out->val = mat_create(rows, cols);

	if (flags & MV_FLAG_REQUIRES_GRAD) { out->grad = mat_create(rows, cols); }

	if (flags & MV_FLAG_INPUT) { model->input = out; }
	if (flags & MV_FLAG_OUTPUT) { model->output = out; }
	if (flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = out; }
	if (flags & MV_FLAG_COST) { model->cost = out; }

	return out;
}

model_var* _mv_unary_impl(
	model_context* model, model_var* input, int rows, int cols,
	int flags, model_var_op op
) {
	if (input->flags & MV_FLAG_REQUIRES_GRAD) {
		flags |= MV_FLAG_REQUIRES_GRAD;
	}

	model_var* out = mv_create(model, rows, cols, flags);

	out->op = op;
	out->inputs[0] = input;

	return out;
}

model_var* _mv_binary_impl(
	model_context* model, model_var* a, model_var* b,
	int rows, int cols, int flags, model_var_op op
) {
	if (a->flags & MV_FLAG_REQUIRES_GRAD||
		b->flags & MV_FLAG_REQUIRES_GRAD
	) {
		flags |= MV_FLAG_REQUIRES_GRAD;
	}

	model_var* out = mv_create(model, rows, cols, flags);

	out->op = op;
	out->inputs[0] = a;
	out->inputs[1] = b;

	return out;
}

model_var* mv_relu(model_context* model, model_var* input, int flags) {
	return _mv_unary_impl(
		model, input,
		input->val->rows, input->val->cols,
		flags, MV_OP_RELU
	);
}

model_var* mv_softmax(model_context* model, model_var* input, int flags) {
	return _mv_unary_impl(
		model, input,
		input->val->rows, input->val->cols,
		flags, MV_OP_SOFTMAX
	);
}

model_var* mv_add(model_context* model, model_var* a, model_var* b, int flags) {
	if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
		return NULL;
	}

	return _mv_binary_impl(
		model, a, b, a->val->rows, a->val->cols, flags, MV_OP_ADD
	);
}

model_var* mv_sub(model_context* model, model_var* a, model_var* b, int flags) {
	if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
		return NULL;
	}

	return _mv_binary_impl(
		model, a, b, a->val->rows, a->val->cols, flags, MV_OP_SUB
	);
}

model_var* mv_matmul(model_context* model, model_var* a, model_var* b, int flags) {
	if (a->val->cols != b->val->rows) {
		return NULL;
	}

	return _mv_binary_impl(
		model, a, b, a->val->rows, b->val->cols, flags, MV_OP_MATMUL
	);
}
model_var* mv_cross_entropy(
	model_context* model, model_var* p, model_var* q,
	int flags
) {
	if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
		return NULL;
	}

	return _mv_binary_impl(
		model, p, q, p->val->rows, p->val->cols, flags, MV_OP_CROSS_ENTROPY
	);
}

model_program model_prog_create(model_context* model, model_var* out_var) {
	bool* visited = calloc(model->num_vars, sizeof(*visited));

	int stack_size = 0;
	int out_size;
	model_var** stack = malloc(model->num_vars * sizeof(*stack));
	model_var** out = malloc(model->num_vars * sizeof(*out));

	stack[stack_size++] = out_var;
	while (stack_size > 0) {
		model_var* cur = stack[--stack_size];

		if (cur->index >= model->num_vars) { continue; }

		if (visited[cur->index]) {
			if (out_size < model->num_vars) {
				out[out_size++] = cur;
			}
			continue;
		}

		visited[cur->index] = true;

		if (stack_size < model->num_vars) {
			stack[stack_size++] = cur;
		}

		int num_inputs = MV_NUM_INPUTS(cur->op);
		for (int i = 0; i < num_inputs; i++) {
			model_var* input = cur->inputs[i];

			if (input->index >= model->num_vars || visited[input->index]) {
				continue;
			}

			for (int j = 0; j < stack_size; j++) {
				if (stack[j] == input) {
					for (int k = j; k < stack_size-1; k++) {
						stack[k] = stack[k+1];
					}
					stack_size--;
				}
			}

			if (stack_size < model->num_vars) {
				stack[stack_size++] = input;
			}
		}
	}

	model_program prog = {
		.size = out_size,
		.vars = malloc(sizeof(model_var*) * out_size)
	};

	memcpy(prog.vars, out, sizeof(model_var*) * out_size);

	free(visited);
	free(stack);
	free(out);

	return prog;
}

void model_prog_compute(model_program* prog) {
	for (int i = 0; i < prog->size; i++) {
		model_var* cur = prog->vars[i];

		model_var* a = cur->inputs[0];
		model_var* b = cur->inputs[1];

		switch (cur->op) {
			case MV_OP_NULL:
			case MV_OP_CREATE: break;

			case _MV_OP_UNARY_START: break;

			case MV_OP_RELU: { mat_relu(cur->val, a->val); } break;
			case MV_OP_SOFTMAX: { mat_softmax(cur->val, a->val); } break;

			case _MV_OP_BINARY_START: break;

			case MV_OP_ADD: { mat_add(cur->val, a->val, b->val); } break;
			case MV_OP_SUB: { mat_sub(cur->val, a->val, b->val); } break;
			case MV_OP_MATMUL: {
				mat_mul(cur->val, a->val, b->val, 1, 0, 0);
			} break;
			case MV_OP_CROSS_ENTROPY: {
				mat_cross_entropy(cur->val, a->val, b->val);
			} break;
		}
	}
}

void model_prog_compute_grads(model_program* prog) {
	for (int i = 0; i < prog->size; i++) {
		model_var* cur = prog->vars[i];

		if ((cur->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD) {
			continue;
		}

		if (cur->flags & MV_FLAG_PARAMETER) {
			continue;
		}

		mat_clear(cur->grad);
	}

	mat_fill(prog->vars[prog->size-1]->grad, 1.0f);

	for (int i = prog->size; i >= 0; i++) {
		model_var* cur = prog->vars[i];

		model_var* a = cur->inputs[0];
		model_var* b = cur->inputs[1];

		int num_inputs = MV_NUM_INPUTS(cur->op);

		if (
			num_inputs == 1 &&
			(a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
		) {
			continue;
		}

		if (
			num_inputs == 2 &&
			(a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
			(b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD
		) {
			continue;
		}

		switch (cur->op) {
			case MV_OP_NULL:
			case MV_OP_CREATE: break;

			case _MV_OP_UNARY_START: break;

			case MV_OP_RELU: { };
			case MV_OP_SOFTMAX: { };

			case _MV_OP_BINARY_START: break;

			case MV_OP_ADD: {
				if (a->flags & MV_FLAG_REQUIRES_GRAD) {
					mat_add(a->grad, a->grad, cur->grad);
				}

				if (b->flags & MV_FLAG_REQUIRES_GRAD) {
					mat_add(b->grad, b->grad, cur->grad);
				}
			} break;
			case MV_OP_SUB: {
				if (a->flags & MV_FLAG_REQUIRES_GRAD) {
					mat_sub(a->grad, a->grad, cur->grad);
				}

				if (b->flags & MV_FLAG_REQUIRES_GRAD) {
					mat_sub(b->grad, b->grad, cur->grad);
				}
			} break;
			case MV_OP_MATMUL: {
				if (a->flags & MV_FLAG_REQUIRES_GRAD) {
					mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
				}

				if (b->flags & MV_FLAG_REQUIRES_GRAD) {
					mat_mul(b->grad, a->grad, cur->val, 0, 1, 0);
				}
			} break;
			case MV_OP_CROSS_ENTROPY: { } break;
		}
	}
}
