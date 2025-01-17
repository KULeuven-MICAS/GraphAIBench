#include "graph.h"
void triangle_count(Graph &g, uint64_t &total);
void triangle_count_compressed(Graph &g, uint64_t &total, vidType num_cached = 0);

int main(int argc,char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph> \n";
    std::cout << "Example: " << argv[0] << " ../../inputs/mico/graph\n";
    abort();
  }
  Graph g;
  g.load_compressed_graph(argv[1]);
  g.print_meta_data();
  vidType num_cached = 0;
  if (argc > 2) num_cached = atoi(argv[2]);

  uint64_t total = 0;
  if (num_cached > 0) g.decompress();
  //triangle_count(g, total);
  //std::cout << "total_num_triangles = " << total << "\n";

  total = 0;
  triangle_count_compressed(g, total, num_cached);
  std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}
