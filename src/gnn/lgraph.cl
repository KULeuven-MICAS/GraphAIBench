#include "lgraph.h"
#include "math_functions.hh"
#include "cl_util.h"

void LearningGraph::alloc_on_device() {
  if (d_rowptr_ && gpu_vsize < num_vertices_) clFree(d_rowptr_);
  if (d_rowptr_ == NULL) {
    d_rowptr_ = clMallocRW(num_vertices_+1);
    gpu_vsize = num_vertices_;
  }
  if (d_colidx_ && gpu_esize < num_edges_) clFree(d_colidx_);
  if (d_colidx_ == NULL) {
    d_colidx_ = clMallocRW(num_edges_);
    gpu_esize = num_edges_;
  }
}

void LearningGraph::alloc_on_device(index_t n) {
  d_rowptr_ = clMallocRW(n+1);
  gpu_vsize = n;
}

void LearningGraph::copy_to_gpu() {
  assert(d_rowptr_);
  assert(d_colidx_);
  clMemcpyH2D(d_rowptr_, num_vertices_+1, (void*) row_start_host_ptr());
  clMemcpyH2D(d_colidx_, num_edges_, (void*) edge_dst_host_ptr());
}

void LearningGraph::compute_edge_data() {
  if (edge_data_ && edata_size < num_edges_) delete[] edge_data_; // graph size may change due to subgraph sampling
  if (edge_data_ == NULL) edge_data_ = new edata_t[num_edges_];
  edata_size = num_edges_;
  #pragma omp parallel for
  for (size_t i = 0; i < num_vertices_; i ++) {
    float c_i = std::sqrt(float(get_degree(i)));
    for (auto e = edge_begin(i); e != edge_end(i); e++) {
      const auto j = getEdgeDst(e);
      float c_j  = std::sqrt(float(get_degree(j)));
      if (c_i == 0.0 || c_j == 0.0) edge_data_[e] = 0.0;
      else edge_data_[e] = 1.0 / (c_i * c_j);
    }
  }
}

void LearningGraph::compute_vertex_data() {
  //std::cout << "Computing vertex data\n";
  if (vertex_data_ && vdata_size < num_vertices_) delete[] vertex_data_; // graph size may change due to subgraph sampling
  if (vertex_data_ == NULL) vertex_data_ = new vdata_t[num_vertices_];
  vdata_size = num_vertices_;
  #pragma omp parallel for
  for (size_t v = 0; v < num_vertices_; v ++) {
    auto degree = get_degree(v);
    float temp = std::sqrt(float_t(degree));
    if (temp == 0.0) vertex_data_[v] = 0.0;
    else vertex_data_[v] = 1.0 / temp;
  }
}