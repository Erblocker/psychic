/*
 Copyright (c) 2017 cgoxopx(Yu).
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

inline double sigmoid(double z){
  return 1.0f / (1.0f + exp(-z));
}

inline double sigmoid_gradient(double a){
  return a * (1.0f - a);
}

inline double relu(double z){
  return z > 0? z : 0;
}

inline double relu_gradient(double a){
  return a > 0? 1 : 0;
}

//tanh is predefined
inline double tanh_gradient(double a){
  return 1.0f - a * a;
}

inline double softrelu(double z){
  return log(1.0f + exp(z));
}

inline double softrelu_gradient(double a){
  return 1.0f - exp(-a);
}

inline double leakyrelu(double z){
  return z > 0? z : 0.25f * z;
}

inline double leakyrelu_gradient(double a){
  return a > 0? 1 : 0.25f;
}

inline double linear_regression(double y, double label){
  double delta = y - label;
  return delta * delta;
}

inline double linear_regression_gradient(double y, double label){
  return y - label;
}

//softmax is calculated by pow(z) / sum of pow
inline double negative_log_likelihood_gradient(double a, bool i_equal_j){
  return i_equal_j? a - 1.0f : a;
}


kernel void LSTMCellffd(
  global double * candidate,
  global double * input_gate,
  global double * output_gate,
  global double * forget_gate,
  const global double * cell_candidate_weights,
  const global double * cell_input_weights,
  const global double * cell_output_weights,
  const global double * cell_forget_weights,
  const global double * pre
  ){
    int i=get_global_id(1);
    int j=get_global_id(0);
    double a = pre[j];
    
    atomic_add(&candidate[j]);   //candidate[j]   += (a * cell_candidate_weights[i]);
    atomic_add(&input_gate[j]);  //input_gate[j]  += (a * cell_input_weights[i]);
    atomic_add(&output_gate[j]); //output_gate[j] += (a * cell_output_weights[i]);
    atomic_add(&forget_gate[j]); //forget_gate[j] += (a * cell_forget_weights[i]);
    
}