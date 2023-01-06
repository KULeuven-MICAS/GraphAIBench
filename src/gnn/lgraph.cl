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
