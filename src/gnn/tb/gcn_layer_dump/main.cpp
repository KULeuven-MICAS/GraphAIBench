#include "graph_conv_layer.h"
#include "lgraph.h"
#include <random>
#include <fstream>

#ifdef VORTEX
#include "cl_utils.h"
#endif

std::map<char,double> time_ops;
bool evaluate = false;

int main (int argc, char *argv[]) {
    std::cout << "test gcn_layer" << std::endl;
    int id = 1;
    int nv = 10;
    int din = 5;
    int dout = 10;
    Graph *g = new Graph();
    //TODO add initialization of the graph
    bool act = false;
    float lr = 0.1;
    float feat_drop = 0.1;
    float score_drop = 0.1;
    std::string feats_drop_file = "test.dat";
    std::cout << "test init" << std::endl;

#ifdef VORTEX
    clInit();
#endif    
    GCN_layer gcn_layer(id, nv, din, dout, g, act, lr, feat_drop, score_drop, feats_drop_file);
#ifndef VORTEX
    float *feat_out = new float[nv*dout];
    for (int i = 0; i < nv*din; i++) {
        std::cout << "set input:" << i << std::endl;
        gcn_layer.get_feat_in()[i] = float(rand()/RAND_MAX);
    }
#else
    float feat_in[nv*din];
    for (int i = 0; i < nv*din; i++) {
        feat_in[i] = float(rand()/RAND_MAX);
    }
    std::cout << "Copying inputs..." << std::endl;
    clMemcpyH2D((cl_mem) gcn_layer.get_feat_in(), nv*din*sizeof(float), &feat_in[0]);
    float *feat_out = (float*) clMallocRW(nv*dout*sizeof(float));
#endif

    std::cout << "test forward" << std::endl;
    gcn_layer.forward(&feat_out[0]);

    //test
    std::ifstream ifs(feats_drop_file, std::ios::in | std::ios::binary);
    float B[1];
    for(int i = 0; i < nv*dout; i++) {
        ifs.read((char*)&B[0], sizeof(float));
        if (feat_out[i] != B[0]) {
            std::cout << "ERROR" << std::endl;
            return 1;
        }
    }
    ifs.close();
    std::cout << "SUCCESS" << std::endl;
    return 0;
}