/*
 Copyright (c) 2017 cgoxopx.
 All rights reserved.
 
 Redistribution and use in source and binary forms are permitted
 provided that the above copyright notice and this paragraph are
 duplicated in all such forms and that any documentation,
 advertising materials, and other materials related to such
 distribution and use acknowledge that the software was developed
 by the copyright holder. The name of the
 copyright holder may not be used to endorse or promote products derived
 from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */
 
#define _gradient(z) 1.0f

inline float sigmoid(float z){
  return 1.0f / (1.0f + exp(-z));
}

inline float sigmoid_gradient(float a){
  return a * (1.0f - a);
}

inline float relu(float z){
  return z > 0? z : 0;
}

inline float relu_gradient(float a){
  return a > 0? 1 : 0;
}

//tanh is predefined
inline float tanh_gradient(float a){
  return 1.0f - a * a;
}

inline float softrelu(float z){
  return log(1.0f + exp(z));
}

inline float softrelu_gradient(float a){
  return 1.0f - exp(-a);
}

inline float leakyrelu(float z){
  return z > 0? z : 0.25f * z;
}

inline float leakyrelu_gradient(float a){
  return a > 0? 1 : 0.25f;
}

inline float linear_regression(float y, float label){
  float delta = y - label;
  return delta * delta;
}

inline float linear_regression_gradient(float y, float label){
  return y - label;
}

//softmax is calculated by pow(z) / sum of pow
inline float negative_log_likelihood_gradient(float a, bool i_equal_j){
  return i_equal_j? a - 1.0f : a;
}


kernel void LSTMCellffd(
  global float * candidate,
  global float * input_gate,
  global float * output_gate,
  global float * forget_gate,
  const global float * cell_candidate_weights,
  const global float * cell_input_weights,
  const global float * cell_output_weights,
  const global float * cell_forget_weights,
  const global float * pre
  ){
    int i=get_global_id(1);
    int j=get_global_id(0);
    float a = pre[0];
    
    candidate[j]   += (a * cell_candidate_weights[i]);
    input_gate[j]  += (a * cell_input_weights[i]);
    output_gate[j] += (a * cell_output_weights[i]);
    forget_gate[j] += (a * cell_forget_weights[i]);
    
}