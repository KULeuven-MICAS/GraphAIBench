#include "aggregator.h"
//#include "graph_operations.h" --> checkout what warp reduce does!
#include "cl_utils.h"

void GCN_Aggregator::init(int l, int nv, int ne, float lr, float drop_rate) {
  length = l;
  if (nv > n) {
    if (temp) clFree((cl_mem)temp);
    temp = clMallocRW(nv * l); // avoid repetitive allocation
  }
  n = nv;
}

void GCN_Aggregator::aggregate(int len, Graph& g, const float* in, float* out) {
  unsigned n = g.size();
  auto nnz = g.sizeEdges();
  clSpmm(n, len, n, nnz, g.edge_data_ptr(), (int*)g.row_start_ptr(), (int*)g.edge_dst_ptr(), in, out, temp);
}

void GCN_Aggregator::d_aggregate(int len, Graph& g, const float*, const float* in, float* out) {
    std::cout << "Not implemented yet" << std::endl;
    throw("Too bad!"); 
}