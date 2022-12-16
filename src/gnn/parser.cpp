#include "parser.h"

static void base_show_usage(std::string program_name){
    std::cout << "Usage:" << program_name << "-d data_path" << std::endl;
}

void GNNTrainingParser::show_usage(){
    base_show_usage("$HOME/bin/train");
    std::cout   <<  "\t-e num_epochs(50)"       << std::endl;
    std::cout   <<  "\t-l num_layers(2)"        << std::endl;
    std::cout   <<  "\t-t num_threads(16)"      << std::endl;
    std::cout   <<  "\t-s is_sigmoid(false)"    << std::endl;
    std::cout   <<  "\t-u use_dense(false)"     << std::endl;
    std::cout   <<  "\t-n use_l2norm(false)"    << std::endl;
    std::cout   <<  "\t-f feat_drop(0.)"        << std::endl;
    std::cout   <<  "\t-o score_drop(0.)"       << std::endl;
    std::cout   <<  "\t-i inductive(false)"     << std::endl;
    std::cout   <<  "\t-h dim_hid(16)"          << std::endl;
    std::cout   <<  "\t-r lrate(0.02)"          << std::endl;
    std::cout   <<  "\t-g subg_size(0)"         << std::endl;
    std::cout   <<  "\t-v eval_interval(50)"    << std::endl;
    std::cout   <<  "\t-? help"                 << std::endl;
    std::cout   << "Example: ./bin/infer -d citeseer" << std::endl;
}

void GNNTrainingParser::init_parser(int argc, char* argv[]){
    if (argc <= 2){ 
        show_usage();
        exit(1);
    }
    int c;
    while ((c = getopt(argc, argv, "d:e::l::su:t::nf::o::ih::r::g::v::?")) != -1) {
        switch (c) {
            case 'd':
                args.dataset_path = std::string(optarg);
                break;
            case 't':
                args.num_threads = atoi(optarg);
                break;
            case 'l':
                args.num_layers = atoi(optarg);
                break;
            case 'e':
                args.num_epochs = atoi(optarg);
                break;
            case 's':
                args.is_sigmoid = true;
                break;
            case 'u':
                args.use_dense = true;
                break;
            case 'n':
                args.use_l2norm = true;
                break;
            case 'f':
                args.feat_drop = atof(optarg);
                break;
            case 'o':
                args.score_drop = atof(optarg);
                break;
            case 'i':
                args.inductive = false;
                break;
            case 'h':
                args.dim_hid = atoi(optarg);
                break;
            case 'r':
                args.lrate = atof(optarg);
                break;
            case 'g':
                args.subg_size = atoi(optarg);
                break;
            case 'v':
                args.eval_interval = atoi(optarg);
                break;
            case '?': {
                show_usage();
                exit(0);
            } break;
            default:
                show_usage();
                exit(-1);
        }
    }
}

void GNNTrainingParser::check_args(){
    assert(args.num_layers >= 2);
}

void GNNTrainingParser::reduce_args(){
    if (args.subg_size > 0 || args.arch == gnn_arch::GAT) args.use_l2norm = true;
    if (args.use_l2norm) args.use_dense = true;
}

std::string bool_string(bool flag){
    std::string ret;
    ret = flag ? "true" : "false";
    return ret;
}

void GNNTrainingParser::parse_args(int argc, char* argv[]){
    init_parser(argc, argv);
	check_args();
    reduce_args();
    printf("Dataset name= %s\n",             args.dataset_path.c_str());
    printf("Number of threads= %d\n",        args.num_threads);
    printf("Number of layers= %d\n",         args.num_layers);
    printf("Number of epochs= %d\n",         args.num_epochs);
    printf("Sigmoid= %s\n",                  bool_string(args.is_sigmoid).c_str());
    printf("Dense layers= %s\n",             bool_string(args.use_dense).c_str());
    printf("L2 Norm= %s\n",                  bool_string(args.use_l2norm).c_str());
    printf("Feature dropout= %f\n",          args.feat_drop);
    printf("Score dropaout= %f\n",           args.score_drop);
    printf("Inductive training= %s\n",       bool_string(args.inductive).c_str());
    printf("Hidden layer dimention= %d\n",   args.dim_hid);
    printf("Learning rate= %f\n",            args.lrate);
    printf("Subgraph size= %d\n",            args.subg_size);
    printf("Evaluation interval= %d\n",      args.eval_interval);
}