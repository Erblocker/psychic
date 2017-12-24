//from clnet
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#ifdef cl_khr_int64_base_atomics
	#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
	#define atom_type ulong
#else
	#define atom_type uint
#endif
#ifdef cl_amd_printf
//	#pragma OPENCL EXTENSION cl_amd_printf: enable
#endif
#ifndef work_group_barrier
	#define work_group_barrier barrier
#endif
#ifndef NULL
	#define NULL 0
#endif

#define _gradient(z) 1.0f

inline float sigmoid(float z)
{
	return 1.0f / (1.0f + exp(-z));
}
inline float sigmoid_gradient(float a)
{
	return a * (1.0f - a);
}

inline float relu(float z)
{
	return z > 0? z : 0;
}
inline float relu_gradient(float a)
{
	return a > 0? 1 : 0;
}

//tanh is predefined
inline float tanh_gradient(float a)
{
	return 1.0f - a * a;
}

inline float softrelu(float z)
{
	return log(1.0f + exp(z));
}
inline float softrelu_gradient(float a)
{
	return 1.0f - exp(-a);
}

inline float leakyrelu(float z)
{
	return z > 0? z : 0.25f * z;
}
inline float leakyrelu_gradient(float a)
{
	return a > 0? 1 : 0.25f;
}

inline float linear_regression(float y, float label)
{
	float delta = y - label;
	return delta * delta;
}
inline float linear_regression_gradient(float y, float label)
{
	return y - label;
}

//softmax is calculated by pow(z) / sum of pow
inline float negative_log_likelihood_gradient(float a, bool i_equal_j)
{
	return i_equal_j? a - 1.0f : a;
}

//Parallel: (sizeof(data))
kernel void activation_sigmoid(global float* data)
{
	const int GID = get_global_id(0);
	data[GID] = sigmoid(data[GID]);
}

//Standard implementation
//Parallel: (batch_size * dim_hidden)
kernel void feed_forward_fully_connected_sigmoid(global float* out, const global float* in, const global float* weight, const global float* bias, 
		local float* tmp, const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int n = GID / dim_hidden;
	const int hidden = GID % dim_hidden;
	const int weight_offset = hidden * dim_in;
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

#pragma unroll
	for (int i = 0; i < dim_in; i++)
		z += weight[weight_offset + i] * in[in_offset + i];
	out[GID] = sigmoid(z);
}

//Standard implementation
//Parallel: max(weight_Out_dim, batch_size) x weight_In_dim
kernel void back_propagate_fully_connected_softrelu_gradient(global float* in_grad, global float* weight_grad,
		global float* bias_grad,	const global float* weight, const global float* in, const global float* out,
		const global float* out_grad, const int dim_out, const int dim_in, const int batch_size, const int is_addon)
{
	const int GID = get_global_id(0);
	const int k = GID % dim_in;
	const int n = GID / dim_in;

	if (n < dim_out) {
		float sum_weight_grad = 0, sum_bias_grad = 0;
		for (int j = 0; j < batch_size; j++) {
			const float in_j = in[j * dim_in + k];
			const float out_grad_j = softrelu_gradient(out[j * dim_out + n]) * out_grad[j * dim_out + n];
			sum_bias_grad += out_grad_j;
			sum_weight_grad += in_j * out_grad_j;
		}
		if (k == 0 && bias_grad != NULL)
			bias_grad[n] += sum_bias_grad;
		weight_grad[n * dim_in + k] += sum_weight_grad;
	}

	if (in_grad != NULL && n < batch_size) {
		float sum_in_grad = 0;
		for (int j = 0; j < dim_out; j++) {
			const float weight_j = weight[j * dim_in + k];
			const float out_grad_j = softrelu_gradient(out[n * dim_out + j]) * out_grad[n * dim_out + j];
			sum_in_grad += weight_j * out_grad_j;
		}
		if (is_addon)
			in_grad[n * dim_in + k] += sum_in_grad;
		else
			in_grad[n * dim_in + k] = sum_in_grad;
	}
}

//Parallel: (batch_size * dim_hidden * get_local_size(0))
//Note: choose local NDRange size near (2 * dim_in) when enqueue ASAP
kernel void feed_forward_fully_connected_softrelu(global float* out, const global float* in, const global float* weight, const global float* bias, 
		local float* tmp, const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
// parallel for reduction: should be power of 2
// Use max parallel get_local_size(0) because OpenCL only synchorizes in a work group 
	const int parallel = get_local_size(0);
	const int n = GID / parallel / dim_hidden;
	const int hidden = (GID / parallel) % dim_hidden;
	const int weight_offset = hidden * dim_in;
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

// parallel addition, trade space for time
	const int pos = GID % parallel;
	float sum = 0;
// support any value for dim_in. Inefficient for dim_in < parallel / 2
	for (int index = pos; index < dim_in; index += parallel)
		sum += weight[weight_offset + index] * in[in_offset + index];

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0)
		out[GID / parallel] = softrelu(z + tmp[0]);
}

// fused version
//Parallel: weight_Out_dim x weight_In_dim
kernel void back_propagate_cascade_fully_connected_sigmoid_gradient(global float* weight_grad, global float* bias_grad, const global float* in,
		const global float* out, const global float* weight_next, const global float* nabla_next/*bias_grad*/,
		local float* tmp, const int dim_out, const int dim_in/*k*/, const int dim_weight_next_out, const int batch_size)
{
	const int GID = get_global_id(0);
	const int j = GID / dim_in;
	const int k = GID % dim_in;
	float this_bias_grad = 0;
	float this_weight_grad = 0;

	for (int n = 0; n < batch_size; n++) {
		float a_j = out[n * dim_out + j];

		float sum = 0;
		for (int i = 0; i < dim_weight_next_out; i++)
			sum += weight_next[i * dim_in + j] * nabla_next[i];

		float nabla = sum * sigmoid_gradient(a_j);
		this_bias_grad += nabla;
		this_weight_grad += in[n * dim_in + k] * nabla;
	}
	if (k == 0 && bias_grad != NULL)
		bias_grad[j] += this_bias_grad;
	weight_grad[GID] += this_weight_grad;
}

//Parallel: (batch_size * get_local_size(0))
kernel void negative_log_likelihood_loss(global float* out_grad, global float* out, const global float* in, 
		const global float* label, local float* tmp, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0); //no more than dim_in
	const int n = GID / parallel;
	const int pos = GID % parallel;

	float max_value = in[n * dim_in];
	for (int index = 1; index < dim_in; index++)
		max_value = max(max_value, in[n * dim_in + index]);

	float sum = 0;
	for (int index = pos; index < dim_in; index += parallel) {
		const int k = n * dim_in + index;
		out[k] = exp(in[k] - max_value);
		sum += out[k];
	}

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	sum = tmp[0];

	for (int index = pos; index < dim_in; index += parallel) {
		const int i = ((int) label[n]), k = n * dim_in + index;
		out[k] /= sum;
		out_grad[k] = negative_log_likelihood_gradient(out[k], index == i);
	}
}

//Parallel: (batch_size * get_local_size(0))
kernel void linear_regression_loss(global float* out_grad, const global float* out, const global float* label)
{
	const int GID = get_global_id(0);
	out_grad[GID] = linear_regression_gradient(out[GID], label[GID]);
}

// fused version
//Parallel: (weight_Out_dim * weight_In_dim * get_local_size(0))
kernel void back_propagate_linear_regression_loss_softrelu_gradient(global float* weight_grad, global float* bias_grad, const global float* in,
		const global float* out, const global float* label, local float* tmp/*size is 2 * get_local_size(0)*/,
		const int dim_out/*num_hidden*/, const int dim_in/*k*/, const int batch_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int j = GID / parallel / dim_in;
	const int k = (GID / parallel) % dim_in;
	const int pos = GID % parallel;

	float nabla_sum = 0, weight_grad_sum = 0;
	for (int n = pos; n < batch_size; n += parallel) {
		const float a_j = out[n * dim_out + j];
		float nabla = linear_regression_gradient(a_j, label[n * dim_out + j]) * softrelu_gradient(a_j);
		nabla_sum += nabla;
		weight_grad_sum += in[n * dim_in + k] * nabla;
	}

	local float* tmp_w = tmp + parallel;
	tmp[pos] = nabla_sum;
	tmp_w[pos] = weight_grad_sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride) {
			tmp[pos] += tmp[pos + stride];
			tmp_w[pos] += tmp_w[pos + stride];
		}
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0) {
		if (k == 0 && bias_grad != NULL)
			bias_grad[j] += tmp[0];
		weight_grad[GID / parallel] += tmp_w[0];
	}
}

// fused version for softmax TODO: compute out(j)=out(j)/sigma(out(i)) firstly
//Parallel: (weight_Out_dim * weight_In_dim * get_local_size(0))
kernel void back_propagate_negative_log_likelihood_loss_softrelu_gradient(global float* weight_grad, global float* bias_grad, const global float* in,
		const global float* out, const global float* label, local float* tmp/*size is 2 * get_local_size(0)*/,
		const int dim_out/*num_hidden*/, const int dim_in/*k*/, const int batch_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int j = GID / parallel / dim_in;
	const int k = (GID / parallel) % dim_in;
	const int pos = GID % parallel;

	float nabla_sum = 0, weight_grad_sum = 0;
	for (int n = pos; n < batch_size; n += parallel) {
		const float a_j = out[n * dim_out + j];
		const int i = (int) label[n * dim_out + j];
		const bool i_equal_j = i == j;

		const float nabla = negative_log_likelihood_gradient(a_j, i_equal_j) * softrelu_gradient(a_j);
		nabla_sum += nabla;
		weight_grad_sum += in[n * dim_in + k] * nabla;
	}

	local float* tmp_w = tmp + parallel;
	tmp[pos] = nabla_sum;
	tmp_w[pos] = weight_grad_sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride) {
			tmp[pos] += tmp[pos + stride];
			tmp_w[pos] += tmp_w[pos + stride];
		}
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0) {
		if (k == 0 && bias_grad != NULL)
			bias_grad[j] += tmp[0];
		weight_grad[GID / parallel] += tmp_w[0];
	}
}

//Parallel: params->dims
kernel void update_parameters_by_stochastic_gradient_descent(global float* params, global float* params_grad,
		float learning_rate, float weight_decay)
{
	const int GID = get_global_id(0);
	params[GID] -= learning_rate * (params_grad[GID] + weight_decay * params[GID]);
	params_grad[GID] = 0;
}

//Parallel: (h_batch_size * h_dim_hidden)
kernel void feed_forward_LSTM_recurrent(global float* x_timestep, global float* out, const global float* x, const global float* hidden,
		const int timestep, const int sequence_length, const int dim_input, const int dim_hidden)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int j = GID % dim_hidden;
	const int offset = batch * sequence_length + abs(timestep);
	// fetch input timestep from batch-major data
	if (timestep >= 0) { //exclude the last call: need not generate x_timestep
		const int m = batch * dim_input, n = offset * dim_input;
		for (int index = j; index < dim_input; index += dim_hidden)
			x_timestep[m + index] = x[n + index]; //collect timestep from batch-major data
	}

	//save hidden result to out
	if (out != NULL) { //exclude the first call: no hidden output at this time
		const int k = (offset - 1) * dim_hidden + j;
		out[k] = hidden[GID];
	}
}

//Parallel: (h_batch_size * h_dim_hidden)
kernel void back_propagate_LSTM_recurrent(global float* hidden_grad, global float* x_grad, global float* x_timestep, const global float* out_grad, 
		const global float* x_timestep_grad, const global float* x, const int timestep, const int sequence_length, const int dim_input, const int dim_hidden)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int j = GID % dim_hidden;
	const int offset = batch * sequence_length + abs(timestep);
	//save hidden result from out_grad
	if (out_grad != NULL) { //exclude the first call: no hidden output at this time
		const int k = (offset - 1) * dim_hidden + j;
		hidden_grad[GID] += out_grad[k]; //add on back-propagation-through-time gradient
	}

	// put input gradient as batch-major data
	if (timestep > 0) { //exclude the last call: need not generate x_timestep_grad
		const int m = batch * dim_input, n = offset * dim_input;
		for (int index = j; index < dim_input; index += dim_hidden) {
			const int i = m + index, k = n + index;
			x_timestep[i] = x[k - dim_input]; //recover input
			x_grad[k] = x_timestep_grad[i]; //restore batch-major data from timestep data
		}
	}
	else if (timestep == 0) { //x_timestep is ready in the first time, and need not to be prepared in the last time, so both ignored)
		const int m = batch * dim_input, n = offset * dim_input;
		for (int index = j; index < dim_input; index += dim_hidden)
			x_grad[n + index] = x_timestep_grad[m + index]; //restore batch-major data from timestep data
	}
}

//Parallel: (batch_size * dim_hidden)
kernel void feed_forward_LSTM_cell(global float* C, global float* h, global float* gates_data/*cell_no_max * batch_size * 5*dim_hidden*/,
		const global float* z/*batch_size * 4*dim_hidden*/, local float* tmp, const int dim_hidden, int cell_no)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int i_g = batch * 4 * dim_hidden + (GID % dim_hidden);
	const int i_t = i_g + dim_hidden;
	const int f_g = i_t + dim_hidden;
	const int o_g = f_g + dim_hidden;
	const float in_gate= sigmoid(z[i_g]);
	const float C_candicate = tanh(z[i_t]);
	const float forget_gate = sigmoid(z[f_g]);
	const float out_gate = sigmoid(z[o_g]);
	const float C_prev = C[GID]; //initialized as zero for first timestamp
	const float C_t = forget_gate * C_prev + in_gate * C_candicate;
	const float tanh_C_t = tanh(C_t);

	if (gates_data != NULL) {
		global float* data = gates_data + cell_no * get_global_size(0) * 7;

		const float C_grad = out_gate * tanh_gradient(tanh_C_t);
		data[i_g] = C_candicate * sigmoid_gradient(in_gate);
		data[i_t] = in_gate * tanh_gradient(C_candicate);
		data[f_g] = C_prev * sigmoid_gradient(forget_gate);
		data[o_g] = tanh_C_t * sigmoid_gradient(out_gate);

		const int p = 4 * get_global_size(0) + GID;
		const int c_g = p + get_global_size(0);
		const int c_m = c_g + get_global_size(0);
		data[p] = h[GID]; //h_prev: initialized as zero for first timestamp
		data[c_g] = C_grad;
		data[c_m] = forget_gate;
	}
	C[GID] = C_t;
	h[GID] = out_gate * tanh_C_t;
}

//Parallel: (batch_size * dim_hidden)
kernel void back_propagate_LSTM_cell_gates(global float* z_grad, global float* h_prev, global float* cell_state_grad, const global float* h_grad,
		const global float* gates_data, local float* tmp, const int dim_hidden, const int batch_size, const int cell_no)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int i_g = batch * 4 * dim_hidden + (GID % dim_hidden);
	const int i_t = i_g + dim_hidden;
	const int f_g = i_t + dim_hidden;
	const int o_g = f_g + dim_hidden;
	const int p = 4 * get_global_size(0) + GID;
	const int c_g = p + get_global_size(0);
	const int c_m = c_g + get_global_size(0);

	const global float* data = gates_data + cell_no * get_global_size(0) * 7;
	const float h_grad_batch_one = h_grad[GID];
	const float C_grad = data[c_g];
	const float forget_gate = data[c_m];
	const float cell_grad = h_grad_batch_one * C_grad + cell_state_grad[GID];

	z_grad[i_g] = cell_grad * data[i_g];
	z_grad[i_t] = cell_grad * data[i_t];
	z_grad[f_g] = cell_grad * data[f_g];
	z_grad[o_g] = h_grad_batch_one * data[o_g];
	h_prev[GID] = data[p];
	cell_state_grad[GID] = cell_grad * forget_gate;
}

//Parallel: (dim_hidden * get_local_size(0))
kernel void back_propagate_fully_connected_LSTM_cell(global float* weight_h_grad, global float* weight_x_grad, global float* bias_grad,
		global float* h_grad, global float* x_grad, const global float* gates_data,const global float* x, const global float* weight_h,
		const global float* weight_x, local float* tmp, const int dim_input, const int dim_hidden, const int batch_size, const int cell_no)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int j = GID / parallel;
	const int K = GID % parallel;
	const int i_g = j; //0 <= i_g <= dim_hidden
	const int i_t = i_g + dim_hidden;
	const int f_g = i_t + dim_hidden;
	const int o_g = f_g + dim_hidden;
	gates_data += cell_no * dim_hidden * batch_size * 7;

	const global float* data = gates_data;
	for (int k = K; k < dim_hidden; k += parallel) //folad for hidden input vector size (dim_hidden)
		for (int n = 0; n < batch_size; n++) {
			const float h_prev = data[4 * dim_hidden + k];
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			weight_h_grad[i_g * dim_hidden + k] += h_grad_batch_one * data[i_g] * h_prev;
			weight_h_grad[i_t * dim_hidden + k] += h_grad_batch_one * data[i_t] * h_prev;
			weight_h_grad[f_g * dim_hidden + k] += h_grad_batch_one * data[f_g] * h_prev;
			weight_h_grad[o_g * dim_hidden + k] += h_grad_batch_one * data[o_g] * h_prev;
			data += dim_hidden * 5;
		}

	data = gates_data;
	for (int k = K; k < dim_input; k += parallel)
		for (int n = 0; n < batch_size; n++) {
			const float in = x[n * dim_input + k];
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			weight_x_grad[i_g * dim_input + k] += h_grad_batch_one * data[i_g] * in;
			weight_x_grad[i_t * dim_input + k] += h_grad_batch_one * data[i_t] * in;
			weight_x_grad[f_g * dim_input + k] += h_grad_batch_one * data[f_g] * in;
			weight_x_grad[o_g * dim_input + k] += h_grad_batch_one * data[o_g] * in;
			data += dim_hidden * 5;
		}

	data = gates_data;
	if (K == 0)
		for (int n = 0; n < batch_size; n++) {
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			bias_grad[i_g] += h_grad_batch_one * data[i_g];
			bias_grad[i_t] += h_grad_batch_one * data[i_t];
			bias_grad[f_g] += h_grad_batch_one * data[f_g];
			bias_grad[o_g] += h_grad_batch_one * data[o_g];

			for (int k = 0; k < dim_hidden; k++) //TODO: wait for parallelizing on local space: parallel reduce for sum
				h_grad[n * dim_hidden + j] += h_grad_batch_one * (data[k] + data[k + dim_hidden] + data[k + 2 * dim_hidden] + data[k + 3 * dim_hidden]) * weight_h[k * dim_hidden + j];
			data += dim_hidden * 5;
		}

	data = gates_data;
	for (int k = K; k < dim_input; k += parallel)
		for (int n = 0; n < batch_size; n++) {
			const float weight = weight_x[n * dim_input + k];
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			x_grad[i_g * dim_input + k] = h_grad_batch_one * data[i_g] * weight_x[i_g * dim_input + k];
			x_grad[i_t * dim_input + k] = h_grad_batch_one * data[i_t] * weight_x[i_t * dim_input + k];
			x_grad[f_g * dim_input + k] = h_grad_batch_one * data[f_g] * weight_x[f_g * dim_input + k];
			x_grad[o_g * dim_input + k] = h_grad_batch_one * data[o_g] * weight_x[o_g * dim_input + k];
			data += dim_hidden * 5;
		}
}

kernel void parallel_add(global float* z, const global float* a, const global float* b)
{
	const int GID = get_global_id(0);
	z[GID] = a[GID] + b[GID];
}

kernel void parallel_plus(global float* a, const global float* b)
{
	const int GID = get_global_id(0);
	a[GID] += b[GID];
}

//Parallel: (batch_size * dim_hidden * get_local_size(0))
kernel void parallel_multiply(global float* out, const global float* weight, const global float* in, local float* tmp, const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int k = GID / parallel;
	const int n = k / dim_hidden;
	const int hidden = k % dim_hidden;
	const int weight_offset = hidden * dim_in;
	const int in_offset = n * dim_in;
	float z = 0;

	const int pos = GID % parallel;
	float sum = 0;
	for (int index = pos; index < dim_in; index += parallel)
		sum += weight[weight_offset + index] * in[in_offset + index];

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0)
		out[k] = softrelu(z + tmp[0]);
}

//Parallel: (batch_size * out_height * out_width * out_depth)
kernel void feed_forward_convolution_activation_relu(global float* out, const global float* weight/*out_depth * kernel_height * kernel_width * in_depth*/, const global float* bias,
		const global float* in/*batch_size * in_height * in_width * in_depth*/, const int in_height, const int in_width, const int in_depth,
		const int kernel_height, const int kernel_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	const int out_height = get_global_size(1);
	const int out_width = get_global_size(2);
	const int out_depth = get_global_size(0) / batch_size;
	const int n = get_global_id(0) / out_depth; //batch
	const int rout = get_global_id(1);
	const int cout = get_global_id(2);
	const int filter = get_global_id(0) % out_depth;
	const int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;

	float sum = bias != NULL? bias[filter] : 0;
	// convolution operation for the image locations centered at (rout, cout)
	for (int kr = 0; kr < kernel_height; kr++)
		for (int kc = 0; kc < kernel_width; kc++) {
			int rin = rout * stride_height + kr - padding_height;
			int cin = cout * stride_width + kc - padding_width;
			if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width)
				continue;
			int weight_index = ((filter * kernel_height + kr) * kernel_width + kc) * in_depth;
			int in_index = ((n * in_height + rin) * in_width + cin) * in_depth;
#ifdef CONVOLUTION_VECTOR
			int channel = 16;
			float16 sum16 = 0;
			for (; channel <= in_depth; channel += 16, weight_index += 16, in_index += 16)
				sum16 += (*(global float16*) (weight + weight_index)) * (*(global float16*) (in + in_index)); //error CL_OUT_OF_RESOURCES on NVIDIA GTX1050Ti, driver version 382.05
			float8 sum8 = sum16.lo + sum16.hi;
			float4 sum4 = sum8.lo + sum8.hi;
			float2 sum2 = sum4.lo + sum4.hi;
			sum += sum2.lo + sum2.hi;
			for (channel -= 16; channel < in_depth; channel++) //cross channel
				sum += weight[weight_index++] * in[in_index++];
#else
			for (int channel = 0; channel < in_depth; channel++) //cross channel
				sum += weight[weight_index++] * in[in_index++];
#endif
		}
	out[offset] = relu(sum);
}

//Parallel: (out_depth * kernel_height * kernel_width * in_depth)
kernel void back_propagate_convolution_relu_gradient(global float* weight_grad/*out_depth * kernel_height * kernel_width * in_depth*/,
		global float* bias_grad/*out_depth*/, const global float* in, const global float* out,
		const global float* out_grad, const int in_height, const int in_width, const int in_depth,
		const int out_height, const int out_width, const int out_depth, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	const int kernel_height = get_global_size(1);
	const int kernel_width = get_global_size(2);
	const int filter = get_global_id(0) / in_depth;
	const int kr = get_global_id(1);
	const int kc = get_global_id(2);
	const int kd = get_global_id(0) % in_depth;

	const int GID = ((filter * kernel_height + kr) * kernel_width + kc) * in_depth + kd;
	float sum_weight_grad = 0, sum_bias_grad = 0;
	int in_offset = kd;
	int out_offset = filter;
	for (int n = 0; n < batch_size; n++, in_offset += in_height * in_width * in_depth, out_offset += out_height * out_width * out_depth)
		for (int rout = 0; rout < out_height; rout++) {
			int rin = rout * stride_height + kr - padding_height;
			if (rin < 0 || rin >= in_height)
				continue;
			for (int cout = 0; cout < out_width; cout++) {
				int cin = cout * stride_width + kc - padding_width;
				if (cin < 0 || cin >= in_width)
					continue;
				int in_index = in_offset + (rin * in_width + cin) * in_depth;
				int out_index = out_offset + (rout * out_width + cout) * out_depth;
				float out_gradient = out_grad[out_index];
				float func_grad = relu_gradient(out[out_index]);
				float data = in[in_index];
				float gradient = func_grad * out_gradient;
				sum_bias_grad += gradient;
				sum_weight_grad += gradient * data;
			}
		}
	weight_grad[GID] += sum_weight_grad;
	if (kr == 0 && kc == 0 && kd == 0 && bias_grad != NULL)
		bias_grad[filter] += sum_bias_grad;
}

//Parallel: (batch_size * in_height * in_width * in_depth)
kernel void back_propagate_convolution_relu_gradient_for_input(global float* in_grad, const global float* weight, const global float* out,
		const global float* out_grad, const int kernel_height, const int kernel_width, const int in_depth,
		const int out_height, const int out_width, const int out_depth, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	const int in_height = get_global_size(1);
	const int in_width = get_global_size(2);
	const int n = get_global_id(0) / in_depth; //batch
	const int rin = get_global_id(1);
	const int cin = get_global_id(2);
	const int channel = get_global_id(0) % in_depth;

	const int GID = ((n * in_height + rin) * in_width + cin) * in_depth + channel;
	float sum_in_grad = 0;
	const int kernel_volume = kernel_height * kernel_width * in_depth;
	for (int kr = 0; kr < kernel_height; kr++)
		for (int kc = 0; kc < kernel_width; kc++) {
			int rout = (rin - kr + padding_height) / stride_height;
			int cout = (cin - kc + padding_width) / stride_width;
			if (rout < 0 || rout >= out_height || cout < 0 || cout >= out_width)
				continue;
			int out_index = ((n * out_height + rout) * out_width + cout) * out_depth;
			int weight_index = (kr * kernel_width + kc) * in_depth + channel;
			for (int filter = 0; filter < out_depth; filter++, weight_index += kernel_volume) {
				float out_gradient = out_grad[out_index];
				float func_grad = relu_gradient(out[out_index++]);
				float factor = weight[weight_index];
				sum_in_grad += func_grad * out_gradient * factor;
			}
		}
	in_grad[GID] = sum_in_grad;
}

//Parallel: (batch_size * out_height * out_width * out_depth)
kernel void feed_forward_max_pooling(global float* out, const global float* in/*batch_size * in_height * in_width * in_depth*/,
		const int in_height, const int in_width, const int in_depth/*equals to out_depth*/,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size, global int* out_index)
{
	const int out_height = get_global_size(1);
	const int out_width = get_global_size(2);
	const int out_depth = in_depth;
	const int n = get_global_id(0) / out_depth; //batch
	const int rout = get_global_id(1);
	const int cout = get_global_id(2);
	const int filter = get_global_id(0) % out_depth;
	const int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;

	float max = -FLT_MAX, max_index;
	for (int pr = 0; pr < pool_height; pr++)
		for (int pc = 0; pc < pool_width; pc++) {
			int rin = rout * stride_height + pr - padding_height;
			int cin = cout * stride_width + pc - padding_width;
			if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width) {
				if (max < 0) {
					max = 0;
					max_index = -1;
				}
				continue;
			}
			int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + filter; //channel==filter
			if (in[in_index] > max) {
				max = in[in_index];
				max_index = in_index;
			}
		}
	out[offset] = max;
	out_index[offset] = max_index;
}

//Parallel: (batch_size * in_height * in_width * in_depth)
kernel void back_propagate_max_pooling(global float* in_grad, const global float* out_grad,
		const int out_height, const int out_width, const int out_depth/*equals to in_depth*/,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size, const global int* max_index)
{
	const int in_height = get_global_size(1);
	const int in_width = get_global_size(2);
	const int in_depth = out_depth;
	const int n = get_global_id(0) / in_depth; //batch
	const int rin = get_global_id(1);
	const int cin = get_global_id(2);
	const int channel = get_global_id(0) % in_depth;
	const int global_size = in_height * in_width * in_depth;
	const int offset = (rin * in_width + cin) * in_depth + channel;

	float gradient = 0;
	int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + channel;
	for (int pr = 0; pr < pool_height; pr++)
		for (int pc = 0; pc < pool_width; pc++) {
			int rout = (rin - pr + padding_height) / stride_height;
			int cout = (cin - pc + padding_width) / stride_width;
			if (rout < 0 || rout >= out_height || cout < 0 || cout >= out_width)
				continue;
			int out_index = ((n * out_height + rout) * out_width + cout) * out_depth + channel; //filter==channel
			int index = (int) max_index[out_index];
			gradient += in_index == index? out_grad[out_index] : 0;
		}
	in_grad[in_index] = gradient;
}

//Parallel: (out_height * out_width * out_depth)
kernel void feed_forward_average_pooling(global float* out, const global float* in/*batch_size * depth * in_height * in_width*/,
		local float* tmp, const int depth, const int in_height, const int in_width,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	int out_height = (in_height + padding_height * 2 - pool_height + 1) / stride_height;
	int out_width = (in_width + padding_width * 2 - pool_width + 1) / stride_width;
	int GID = get_global_id(0);
	int filter = GID / out_height / out_width;
	int rout = GID / filter / out_width;
	int cout = (GID / filter) % out_width;

	for (int n = 0; n < batch_size; n++)
		for (int channel = 0; channel < depth; channel++) { //2d pooling  is not cross channel
			float sum = 0;
			for (int pr = 0; pr < pool_height; pr++)
				for (int pc = 0; pc < pool_width; pc++) {
					int rin = rout + pr - padding_height;
					int cin = cout + pc - padding_width;
					if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width)
						continue;
					int in_index = n * depth * in_width * in_height + channel * in_width * in_height + rin * stride_height * in_width + cin * stride_width;
					sum += in[in_index];
				}
			out[n * get_global_size(0) + GID] = sum / pool_height / pool_width;
		}
}

//Parallel: (depth * in_height * in_width)
kernel void back_propagate_average_pooling(global float* in_grad, const global float* out_grad, const global float* out,
		local float* tmp, const int depth, const int in_height, const int in_width,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	int out_height = (in_height + padding_height * 2 - pool_height + 1) / stride_height;
	int out_width = (in_width + padding_width * 2 - pool_width + 1) / stride_width;
	int GID = get_global_id(0);
	int filter = GID / out_height / out_width;
	int rout = GID / filter / out_width;
	int cout = (GID / filter) % out_width;
	
	for (int n = 0; n < batch_size; n++) {
		float gradient = 0;
		int in_index = n * depth * out_width * out_height + rout * in_width + in_height;
		in_grad[in_index] = 0;
		for (int pr = 0; pr < pool_height; pr += stride_height)
			for (int pc = 0; pc < pool_width; pc += stride_width) {
				int rin = rout - pr + padding_height;
				int cin = cout - pc + padding_width;
				if (rin < 0 || rin >= out_height || cin < 0 || cin >= out_width)
					continue;
				int out_index = n * depth * out_width * out_height + filter * out_width * out_height + rin * out_width + cin;
				in_grad[in_index] += out_grad[out_index] / pool_height / pool_width;
			}
	}
}

//kernel sum pooling: omitted. use average pooling instead.

//Parallel: (num_hidden)
kernel void feed_forward_dropout(global float* out, const global float* mask/*num_hidden*/,
		local float* tmp, const int num_hidden, const float p, const int batch_size)
{
	int GID = get_global_id(0);

	for (int n = 0; n < batch_size; n++)
		out[n * get_global_size(0) + GID] *= mask[GID];
}

//Parallel: (num_hidden)
kernel void back_propagate_dropout(global float* out_grad, const global float* mask/*num_hidden*/,
		local float* tmp, const int num_hidden, const float p, const int batch_size, const float max_norm)
{
	int GID = get_global_id(0);

	for (int n = 0; n < batch_size; n++)
		out_grad[n * get_global_size(0) + GID] *= mask[GID];
}

//Parallel: (get_local_size(0) * vector_length)
kernel void feed_forward_embedding(global float* out, global float* input,
		const global float* vector_weight, local float* tmp, const int dim_in, const int vector_length)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int weight_offset = GID / parallel;

	for (int index = GID % parallel; index < dim_in; index += parallel)
		out[index * vector_length + weight_offset] = vector_weight[((int) input[index]) * vector_length + weight_offset];
}

//Parallel: (vector_length)
kernel void back_propagate_embedding(global float* vector_weight_grad, const global float* input,
		const global float* out_grad, local float* tmp, const int dim_in, const int vector_length, const int dim_vector_num)
{
	const int GID = get_global_id(0);
	for (int i = 0; i < dim_in; i++)
		vector_weight_grad[((int) input[i]) * vector_length + GID] += out_grad[i * vector_length + GID];
}
