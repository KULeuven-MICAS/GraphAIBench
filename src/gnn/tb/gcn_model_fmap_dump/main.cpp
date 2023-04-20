#include "net.h"

std::map<char,double> time_ops;
bool evaluate = false;
std::string dump_path = "outputs/";

int main (int argc, char *argv[]) {
    std::cout << "Model test v1" << std::endl;
    Model<GCN_layer> model;
    GNNTrainingParser parser;
    std::cout << "Parsing the arguments..." << std::endl;
    parser.parse_args(argc, argv);
    std::cout << "Loading data..." << std::endl;
    model.load_data(&parser);
    model.set_feats_drop_path(dump_path);
    model.construct_network();
    model.infer();
    std::cout << "Done" << std::endl;
    return 0;
}