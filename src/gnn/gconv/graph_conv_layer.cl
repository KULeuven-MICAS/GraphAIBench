#include "graph_conv_layer.h"
#include "cl_utils.h"

template <typename Aggregator>
graph_conv_layer<Aggregator>::graph_conv_layer(int id, int nv, int din, int dout, 
                 LearningGraph *g, bool act, bool concat, float lr, float feat_drop, float score_drop) :
    level_(id), num_samples(nv), dim_in(din), dim_out(dout), graph(g), is_act(act),
    is_bias(false), use_concat(concat), feat_dropout_rate(feat_drop), score_dropout_rate(score_drop) {

    auto x = num_samples;
    auto y = dim_in;
    auto z = dim_out;
    auto init_range = sqrt(6.0 / (y + z));
    
    d_in_temp = clMallocRW(x * y);
    d_out_temp = clMallocRW(x * z);
    clInitConstMem<float>(x * y, 0.0, d_in_temp);
    clInitConstMem<float>(x * z, 0.0, d_out_temp);
    if (y <= z){
        _clMallocRW(x * y, d_in_temp1);
        clInitConstMem<float>(x * y, 0.0, d_in_temp1);
    }
    if (level_ > 0) {
        _clMallocRW(x * y, feat_in);
        clInitConstMem<float>(x * y, 0.0, feat_in);
    }

    if (is_bias) {
    _clMallocRW(z, d_bias);
    clInitConstMem<float>(z, 0.0, d_bias);
    }
}

template <typename Aggregator>
void graph_conv_layer<Aggregator>::update_dim_size(size_t x) {
    if (x > num_samples) {
        auto y = dim_in;
        auto z = dim_out;
        _clReallocRW(x * y, d_in_temp);
        _clReallocRW(x * z, d_out_temp);
        clInitConstMem<float>(x * y, 0.0, d_in_temp);
        clInitConstMem<float>(x * z, 0.0, d_out_temp);
        if (y <= z){
            _clReallocRW(x * y, d_in_temp1);
            clInitConstMem<float>(x * y, 0.0, d_in_temp1);
        }
        if (level_ > 0) {
            _clReallocRW(x * y, feat_in);
            clInitConstMem<float>(x * y, 0.0, feat_in);
        }
    }
}

template class graph_conv_layer<GCN_Aggregator>;
template class graph_conv_layer<GAT_Aggregator>;
template class graph_conv_layer<SAGE_Aggregator>;