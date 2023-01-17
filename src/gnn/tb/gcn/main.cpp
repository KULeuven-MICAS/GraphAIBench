#include "parser.h"
#include "reader.h"
#include "graph.h"
#include "cl_utils.h"
#include "cl_math_functions.h"

int dim_hid = 16;

int main (int argc, char * argv []) {

    //CPU data
    std::string dataset_path = "/users/micas/gsarda/vortex/vortex-KUL-fork/tests/opencl/GraphAIBanch/datasets/cora";

    auto full_graph = new Graph(false); // true means graph on GPU
    auto reader = new Reader(dataset_path);
    std::vector<float> input_features;

    reader->set_dataset();
    reader->readGraphFromGRFile(full_graph);
    int num_samples = full_graph->size();
    int num_cls = reader->read_labels(NULL, 0); //fake labels read
    full_graph->add_selfloop();
    
    int dim_init = reader->read_features(input_features);

    //construct network here
    size_t layers_size = 2;
    int dim_in[layers_size]      = {dim_init, dim_hid};
    int dim_out[layers_size]     = {dim_hid, num_cls};

    float* in_temp[layers_size];
    float* out_temp[layers_size];
    float* out_temp1[layers_size];
    float* aggr_temp[layers_size];
    float* feat_in[layers_size+1];
    float* W_neigh[layers_size];
    for (int i = 0; i < layers_size; i++) {
        in_temp[i]  = (float*)malloc(num_samples * dim_in[i] * sizeof(float));
        out_temp[i] = (float*)malloc(num_samples * dim_out[i] * sizeof(float));
        W_neigh[i]  = (float*)malloc(dim_in[i] * dim_out[i] * sizeof(float));
        aggr_temp[i] = (float*)malloc(num_samples * dim_out[i] * sizeof(float));
        for (int j = 0; j < num_samples * dim_in[i]; j++) W_neigh[j] = random<float>(-(1.0 / sqrt(dim_in[i] + dim_out[i])), (1.0 / sqrt(dim_in[i] + dim_out[i])));
        if (dim_in[i] <= dim_out[i]) 
            out_temp1[i] = (float*)malloc(num_samples * dim_out[i] * sizeof(float));
        if (i > 0)
            feat_in[i] = (float*)malloc(num_samples * dim_in[i] * sizeof(float));
    }

    //GPU data
    clInit();
    float* d_input_features = clMallocRW(num_samples * dim_init * sizeof(float));
    clMemcpyH2D(d_input_features, input_features.data(), num_samples * dim_init * sizeof(float));

    //generate here partitioning
    //transfer_data_to_device();
    full_graph->alloc_on_device();
    full_graph->copy_to_gpu();
    full_graph->compute_vertex_data();

    float* d_in_temp[layers_size];
    float* d_out_temp[layers_size];
    float* d_out_temp1[layers_size];
    float* d_aggr_temp[layers_size];
    float* d_feat_in[layers_size+1];
    float* d_W_neigh[layers_size];
    struct oclKernelParamStruct* work_groups[layers_size];

    float init_range;
    for (int i = 0; i < layers_size; i++) {
        d_in_temp[i]  = (float*)clMallocRW(num_samples * dim_in[i] * sizeof(float));
        d_out_temp[i] = (float*)clMallocRW(num_samples * dim_out[i] * sizeof(float));
        d_W_neigh[i]  = (float*)clMallocRW(dim_in[i] * dim_out[i] * sizeof(float));
        d_aggr_temp[i] = (float*)clMallocRW(num_samples * dim_out[i] * sizeof(float));
        if (dim_in[i] <= dim_out[i])
            d_out_temp1[i] = clMallocRW(num_samples * dim_out[i] * sizeof(float));
        if (i > 0)
            d_feat_in[i] = clMallocRW(num_samples * dim_in[i] * sizeof(float));
        work_groups[i].global_work_size = (size_t*) malloc(2 * sizeof(size_t));
        work_groups[i].global_work_size[0] = num_samples;
        work_groups[i].global_work_size[1] = dim_in[i];
        work_groups[i].local_work_size = (size_t*) malloc(2 * sizeof(size_t));
        work_groups[i].local_work_size[0] = (num_samples<16) ? 16 : num_samples[i] ;
        work_groups[i].local_work_size[1] = (dim_in[i]<16) ? 16 : dim_in[i] ;
    }
    d_feat_in[0] = d_input_features;
    d_feat_in[2] = clMallocRW(num_samples * num_cls * sizeof(float));

    //forward_layer one
    for (int i = 0; i < layers_size; i++) {
        if (dim_in[i] > dim_out[i]) {
            clMatmul(   work_groups[i], 
                        num_samples[i], 
                        dim_out[i], 
                        dim_in[i], 
                        d_feat_in[i], 
                        d_W_neigh[i], 
                        d_out_temp[i]); // x*y; y*z; x*z
            //aggr.aggregate(dim_out[i], *graph, d_out_temp, feat_out); // x*x; x*z; x*z
            clAvgAggr( work_groups[i],
                        full_graph.size(), 
                        dim_out[i], 
                        (float*) full_graph.edge_data_ptr(), 
                        (int*)full_graph.row_start_ptr(), 
                        (int*)full_graph.edge_dst_ptr(), 
                        d_out_temp, 
                        d_feat_in[i+1], 
                        d_aggr_temp[i]);
        } else {
            clAvgAggr( full_graph.size(), 
                    dim_in[i], 
                    full_graph.size(), 
                    full_graph.sizeEdges(), 
                    full_graph.edge_data_ptr(), 
                    (int*)full_graph.row_start_ptr(), 
                    (int*)full_graph.edge_dst_ptr(), 
                    d_out_temp, 
                    d_feat_in[i+1], 
                    d_aggr_temp[i]);
            //aggr.aggregate(dim_in[0], *graph, in_data, d_in_temp1); // x*x; x*y; x*y
            clMatmul(   work_groups[i],
                        num_samples[i], 
                        dim_out[i], 
                        dim_in[i], 
                        d_in_temp1[i], 
                        d_W_neigh[i], 
                        d_feat_in[i+1]); // x*y; y*z; x*z
        }
    }

    return 0
}

//TODO
// 1. Add support for vortex to lgraph
// 2. Add support for random init graph?
