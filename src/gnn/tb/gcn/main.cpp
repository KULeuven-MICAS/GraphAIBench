#include "parser.h"

void transfer_data_to_device();

int main (int argc, char * argv []) {

    GNNTrainingParser parser;
    std::cout << "Parsing the arguments..." << std::endl;
    parser.parse_args(argc, argv);

    auto full_graph = new Graph(false); // true means graph on GPU
    auto reader = new Reader(parser.args.dataset_path);
    reader->set_dataset();
    reader->readGraphFromGRFile(full_graph);
    num_samples = full_graph->size();
    full_graph->add_selfloop();
    dim_init = reader->read_features(input_features);
    num_cls = reader->read_labels(labels, !is_sigmoid);

    std::vector<mask_t> masks_val;
    //size_t val_begin, val_end, val_count;       // vertex id range for validation set
    //val_count = reader->read_masks("val", num_samples, val_begin, val_end, masks_test.data());
    size_t val_begin, val_end, val_count;       // vertex id range for validation set
    size_t sample_size = 1;
    val_count = reader->read_masks("val", num_samples, val_begin, val_begin+sample_size, masks_test.data()); 

    //generate here partitioning
    transfer_data_to_device();
    full_graph->alloc_on_device();
    full_graph->copy_to_gpu();
    full_graph->compute_vertex_data();
    
    //construct network here
    size_t layers_size = 2;
    
    return 0
}

//TODO
// 1. Add support for vortex to lgraph
// 2. Add support for random init graph?

void transfer_data_to_device() {//implement

}