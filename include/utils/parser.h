#pragma once
#include "global.h"
#include "math_functions.hh"
#include <unistd.h>

template <typename T>
class Parser {
public:
    T args;
    Parser(){};
private:
    static void show_usage() {};
    virtual void init_parser(int argc, char* argv[]) = 0;
    virtual void  check_args() = 0;
};

typedef struct TrainingArgs
{
    std::string dataset_path;
    int num_epochs = 10;
    int num_threads = 2;
    bool is_sigmoid = DEFAULT_IS_SIGMOID;
    bool use_dense = false;
    bool use_l2norm = false;
    gnn_arch arch = gnn_arch::GCN;
  	float feat_drop = 0.;
  	float score_drop = 0.;
    bool inductive = false;
    int dim_hid = DEFAULT_SIZE_HID;
  	float lrate = DEFAULT_RATE_LEARN;
  	int num_layers = DEFAULT_NUM_LAYER;
  	int subg_size = 0;
  	int eval_interval = EVAL_INTERVAL;
} TrainingArgs;

class GNNTrainingParser: public Parser<TrainingArgs> {
public:
	GNNTrainingParser() {};
    void parse_args(int argc, char* argv[]);
private:
    static void show_usage();
	void init_parser(int argc, char* argv[]);
    void set_defaults() {};
    void check_args();
    void reduce_args();
};