#include "aggregator.h"
//#include "graph_operations.h" --> checkout what warp reduce does!
#include "cl_utils.h"

//not really useful... memory allocation is actually done only when fp is done in batches
void GCN_Aggregator::init(int l, int nv, int ne, float lr, float drop_rate) {
  //if (nv > n) {
  //  if (temp) clFree((cl_mem)temp);
  //  temp = (float*) clMallocRW(nv * l); // avoid repetitive allocation
  //}
  n = nv;
  length = l;
}

void GCN_Aggregator::aggregate(int len, Graph& g, const float* in, float* out) {
  unsigned n = g.size();
  clAvgAggr(n, len, (cl_mem) g.edge_data_ptr(), (cl_mem) g.row_start_ptr(), (cl_mem) g.edge_dst_ptr(), (cl_mem) in, (cl_mem) out);
}

void GCN_Aggregator::d_aggregate(int len, Graph& g, const float*, const float* in, float* out) {
    std::cout << "Not implemented yet" << std::endl;
    throw("Too bad!"); 
}