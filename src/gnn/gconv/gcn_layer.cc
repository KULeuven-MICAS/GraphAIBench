#include "cl_utils.h"
#include "cl_math_functions.h"
#include "aggregator.h"
#include "graph_conv_layer.h"

// Assume before calling forward, feat_in has already been filled in. 
// feat_out should be the feat_in of next layer
void GCN_layer::forward(float* feat_out) {
  size_t x = num_samples;
  size_t y = dim_in;
  size_t z = dim_out;
  float* in_data = feat_in;
  
  if (feat_dropout_rate > 0.) {
    std::cout << "Warning. LAYER:" << level_ << "feature dropout not supported." << std::endl;
  }

  if (y > z) {
    clMatmul(x, z, y, (cl_mem) in_data, (cl_mem) d_W_neigh, (cl_mem) d_out_temp, {NULL, NULL}); // x*y; y*z; x*z
    aggr.aggregate(z, *graph, d_out_temp, feat_out); // x*x; x*z; x*z
  } else {
    aggr.aggregate(y, *graph, in_data, d_in_temp1); // x*x; x*y; x*y
    clMatmul(x, z, y, (cl_mem) d_in_temp1, (cl_mem) d_W_neigh, (cl_mem) feat_out, {NULL, NULL}); // x*y; y*z; x*z
  }
  //if (is_bias) bias_mv(x, z, feat_out, d_bias);
  //if (is_act) relu_gpu(x*z, feat_out, feat_out);
}

// Assume before calling backward, grad_in has already been filled in. 
// grad_out should be the grad_in of previous layer, grad_out = d L / d X^(l-1)
void GCN_layer::backward(float* feat_out, float* grad_out) {
    std::cout << "Not implemented yet" << std::endl;
    throw("Too bad!");
}

void GCN_layer::update_weight(optimizer* opt) {
    std::cout << "Not implemented yet" << std::endl;
    throw("Too bad!");
}