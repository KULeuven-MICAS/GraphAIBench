#include "reader.h"
#include "lgraph.h"
#include "cl_utils.h"
#include "cl_math_functions.h"

int dim_hid = 16;
size_t layers_size = 2;


static void aggr(const int vlen, const int vnum, const float *A, const int *A_idx_ptr, const int *A_idx, const float *B, float *C);
static void matmul(int M, int N, int K, const float* A, const float *B, float *C);
static bool almost_equal(float a, float b, int ulp = 21);

int main (int argc, char * argv []) {

    //CPU data
    std::string dataset_path = "../../../../datasets/cora";

    auto full_graph = new Graph(false); // true means graph on GPU
    auto h_full_graph = new Graph(false); // true means graph on GPU
    auto reader = new Reader(dataset_path);
    std::vector<float> input_features;

    reader->set_dataset();
    reader->readGraphFromGRFile(full_graph);
    reader->readGraphFromGRFile(h_full_graph);
    int num_samples = full_graph->size();
    
    num_samples = 32;

    std::vector<label_t> dummy_labels(num_samples, 0);
    int num_cls = reader->read_labels(dummy_labels, 0); //fake labels read
    std::vector<label_t>().swap(dummy_labels); //free memory

    full_graph->add_selfloop();
    
    int dim_init = reader->read_features(input_features);
    dim_init = 32;
    //construct network here
    int dim_in[layers_size]      = {dim_init, dim_hid};
    int dim_out[layers_size]     = {dim_hid, num_cls};

    float* out_temp[layers_size];
    float* feat_in[layers_size+1];
    float* W_neigh[layers_size];
    for (size_t i = 0; i < layers_size; i++) {
        W_neigh[i]  = (float*)malloc(dim_in[i] * dim_out[i] * sizeof(float));
        for (int j = 0; j < dim_out[i] * dim_in[i]; j++) W_neigh[i][j] = random<float>(-(1.0 / sqrt(dim_in[i] + dim_out[i])), (1.0 / sqrt(dim_in[i] + dim_out[i])));
        if (dim_in[i] <= dim_out[i]) 
            out_temp[i] = (float*)malloc(num_samples * dim_in[i] * sizeof(float));
        else
            out_temp[i] = (float*)malloc(num_samples * dim_out[i] * sizeof(float));
        if (i > 0)
            feat_in[i] = (float*)malloc(num_samples * dim_in[i] * sizeof(float));
    }
    feat_in[0] = input_features.data(); 
    feat_in[layers_size] = (float*)malloc(num_samples*num_cls * sizeof(float));

    //GPU data
    clInit();
    std::cout << "Vortex buddy is up and running!" << std::endl;
    std::cout << "Number of samples: " << num_samples << std::endl;
    std::cout << "Number of classes: " << num_cls << std::endl;
    std::cout << "Dimension first layer: " << dim_init << std::endl;
    float* d_input_features = (float*)clMallocRW(num_samples * dim_init * sizeof(float));
    clMemcpyH2D((cl_mem)d_input_features, num_samples * dim_init * sizeof(float), input_features.data());

    //generate here partitioning
    //transfer_data_to_device();
    full_graph->compute_vertex_data();
    full_graph->compute_edge_data();
    h_full_graph->compute_vertex_data();
    h_full_graph->compute_edge_data();

    full_graph->alloc_on_device();
    full_graph->copy_to_gpu();

    float* d_out_temp[layers_size];
    float* d_feat_in[layers_size+1];
    float* d_W_neigh[layers_size];
    struct oclKernelParamStruct work_groups[layers_size];

    float init_range;
    for (int i = 0; i < layers_size; i++) {
        d_out_temp[i] = (float*)clMallocRW(num_samples * dim_out[i] * sizeof(float));
        d_W_neigh[i]  = (float*)clMallocRW(dim_in[i] * dim_out[i] * sizeof(float));
        if (dim_in[i] <= dim_out[i])
            d_out_temp[i] = (float*)clMallocRW(num_samples * dim_in[i] * sizeof(float));
        else
            d_out_temp[i] = (float*)clMallocRW(num_samples * dim_out[i] * sizeof(float));
        if (i > 0)
            d_feat_in[i] = (float*)clMallocRW(num_samples * dim_in[i] * sizeof(float));
        work_groups[i].global_work_size = (size_t*) malloc(2 * sizeof(size_t));
        work_groups[i].global_work_size[0] = num_samples;
        work_groups[i].global_work_size[1] = dim_in[i];
        work_groups[i].local_work_size = (size_t*) malloc(2 * sizeof(size_t));
        work_groups[i].local_work_size[0] = (num_samples<32) ? num_samples : 32;
        work_groups[i].local_work_size[1] = (dim_in[i]<32) ? dim_in[i] : 32;
    }
    d_feat_in[0] = d_input_features;
    d_feat_in[2] = (float*)clMallocRW(num_samples * num_cls * sizeof(float));

    //CPU computation
    //for (int i = 0; i < layers_size; i++) {
    //    if (dim_in[i] > dim_out[i]) {
    //        matmul(num_samples, dim_out[i], dim_in[i], feat_in[i], W_neigh[i], out_temp[i]); // x*y; y*z; x*z
    //        aggr(num_samples, dim_out[i], (float*) h_full_graph->edge_data_ptr(), (int*)h_full_graph->row_start_ptr(), (int*)h_full_graph->edge_dst_ptr(), out_temp[i], feat_in[i+1]); // x*x; x*z; x*z
    //    } else {
    //        aggr(num_samples, dim_in[i], (float*) h_full_graph->edge_data_ptr(), (int*)h_full_graph->row_start_ptr(), (int*)h_full_graph->edge_dst_ptr(), feat_in[i], out_temp[i]); // x*x; x*z; x*z
    //        matmul(num_samples, dim_out[i], dim_in[i], out_temp[i], W_neigh[i], feat_in[i+1]); // x*y; y*z; x*z
    //    }
    //}

    //GPU computation
    //forward_layer 
    cl_mem halo;
    halo = clMallocRW(num_samples * dim_init * sizeof(float));
    for (int i = 0; i < layers_size; i++) {
        if (dim_in[i] > dim_out[i]) {
            clMatmul(   work_groups[i], 
                        num_samples, 
                        dim_out[i], 
                        dim_in[i], 
                        (cl_mem) d_feat_in[i], 
                        (cl_mem) d_W_neigh[i], 
                        (cl_mem) d_out_temp[i]); // x*y; y*z; x*z
            std::cout << "clMatmul done" << std::endl;
            //aggr.aggregate(dim_out[i], *graph, d_out_temp, feat_out); // x*x; x*z; x*z
            clAvgAggr(  work_groups[i],
                        num_samples, 
                        dim_out[i], 
                        (cl_mem) full_graph->edge_data_ptr(),
                        (cl_mem) full_graph->row_start_ptr(), 
                        (cl_mem) full_graph->edge_dst_ptr(), 
                        (cl_mem) d_out_temp, 
                        (cl_mem) d_feat_in[i+1]);
        } else {
            clAvgAggr( work_groups[i], 
                    num_samples, 
                    dim_in[i], 
                    (cl_mem) full_graph->edge_data_ptr(), 
                    (cl_mem)full_graph->row_start_ptr(), 
                    (cl_mem)full_graph->edge_dst_ptr(), 
                    (cl_mem) d_feat_in[i], 
                    (cl_mem) d_out_temp);
            //aggr.aggregate(dim_in[0], *graph, in_data, d_in_temp1); // x*x; x*y; x*y
            clMatmul(   work_groups[i],
                        num_samples, 
                        dim_out[i], 
                        dim_in[i], 
                        (cl_mem) d_out_temp[i], 
                        (cl_mem) d_W_neigh[i], 
                        (cl_mem) d_feat_in[i+1]); // x*y; y*z; x*z
        }
    }
    clFinish(oclHandles.queue);
    //checkup
    float* ref_out_temp[layers_size];
    float* ref_feat_in[layers_size+1];
    for (size_t i = 0; i < layers_size; i++){
        if (dim_in[i] <= dim_out[i]) 
            ref_out_temp[i] = (float*)malloc(num_samples * dim_in[i] * sizeof(float));
        else
            ref_out_temp[i] = (float*)malloc(num_samples * dim_out[i] * sizeof(float));
        ref_feat_in[i]  = (float*)malloc(num_samples * dim_in[i] * sizeof(float));
    }
    ref_feat_in[layers_size] = (float*)malloc(num_samples * num_cls * sizeof(float));
    
    for (size_t i=0; i<layers_size; i++){
        clMemcpyD2H((cl_mem) d_feat_in[i], num_samples * dim_in[i] * sizeof(float), ref_feat_in[i]);
        if (dim_in[i] <= dim_out[i]) 
            clMemcpyD2H((cl_mem) d_out_temp[i], num_samples * dim_in[i] * sizeof(float), ref_out_temp[i]);
        else
            clMemcpyD2H((cl_mem) d_out_temp[i], num_samples * dim_out[i] * sizeof(float), ref_out_temp[i]);
    }

    for (size_t i = 0; i < layers_size; i++) {
        std::cout << "Layer " << i << std::endl;
        std::cout << "ref_feat_in" << std::endl;
        for (int j = 0; j < (num_samples*dim_in[i]); j++) {
    	    if (!almost_equal(feat_in[i][j], ref_feat_in[i][j])) {
                std::cout << "Error at " << j << " " << feat_in[i][j] << " " << ref_feat_in[i][j] << std::endl;
                return 1;
            }
  	    }
        if (dim_in[i] <= dim_out[i]) {
            std::cout << "ref_out_temp" << std::endl;
            for (int j = 0; j < (num_samples*dim_in[i]); j++) {
    	        if (!almost_equal(out_temp[i][j], ref_out_temp[i][j])) {
                    std::cout << "Error at " << j << " " << out_temp[i][j] << " " << ref_out_temp[i][j] << std::endl;
                    return 1;
                }
  	        }
        } else {
            std::cout << "ref_out_temp" << std::endl;
            for (int j = 0; j < (num_samples*dim_out[i]); j++) {
    	        if (!almost_equal(out_temp[i][j], ref_out_temp[i][j])) {
                    std::cout << "Error at " << j << " " << out_temp[i][j] << " " << ref_out_temp[i][j] << std::endl;
                    return 1;
                }
  	        }
        }
    }

    //free
    for (size_t i = 0; i < layers_size; i++){
        free(ref_out_temp[i]);
        free(ref_feat_in[i]);
        free(W_neigh[i]);
        free(feat_in[i]);
        free(out_temp[i]);
    }
    free(ref_feat_in[layers_size]);
    free(feat_in[layers_size]);
    clRelease();
    for (size_t i = 0; i < layers_size; i++){
        clFree((cl_mem)d_W_neigh[i]);
        clFree((cl_mem)d_feat_in[i]);
        clFree((cl_mem)d_out_temp[i]);
    }
    clFree((cl_mem)d_feat_in[layers_size]);
    return 0;
}

//TODO
// 1. Add support for vortex to lgraph
// 2. Add support for random init graph?

static void aggr(const int vlen, const int vnum, const float *A, const int *A_idx_ptr, const int *A_idx, const float *B, float *C){
    vec_t neighbor(vlen);
    for (int i = 0; i < vnum; i++){
        for (int j = A_idx_ptr[i]; j < A_idx_ptr[i+1]; j++){
            for (int k = 0; k < vlen; k++)
                neighbor[k] = B[i*vlen+k] * A[A_idx[j]];
            for (int k = 0; k < vlen; k++)
                C[i*vlen+k] += neighbor[k];
        }
    }
}

static void matmul(int M, int N, int K, const float* A, const float *B, float *C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
          acc += A[k * M + m] * B[n * K + k];
      }
      C[n * M + m] = acc;
    }
  }
}

static bool almost_equal(float a, float b, int ulp /*= 21*/) {
    union fi_t { int i; float f; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    return std::abs(fa.i - fb.i) <= ulp;
}