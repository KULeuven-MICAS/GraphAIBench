#CXX=g++

INCLUDES=-I../../../../include/utils -I../../../../include/gnn -I../../../../include -I../../../../include/layers
CFLAGS= $(INCLUDES) -I$(OPENBLAS_DIR)/include -L$(OPENBLAS_DIR)/lib -fopenmp -pthread -Wall --std=c++11 -lboost_thread -lboost_system -lopenblas -fsanitize=address

VPATH += ../../../common
VPATH += ../../../utilities
VPATH += ../../../layers
VPATH += ../../gconv
VPATH += ../../

COMMON_CXXOBJS = graph_conv_layer.o lgraph.o random.o net.o parser.o reader.o loss_layer.o sampler.o 
GNN_LAYER_CXXOBJS = gcn_layer.o gcn_aggregator.o gat_layer.o gat_aggregator.o sage_layer.o sage_aggregator.o
UTIL_CXXOBJS = math_functions.o optimizer.o
LAYER_OBJS = l2norm_layer.o dense_layer.o
LAYER_CXXOBJS = softmax_loss_layer.o sigmoid_loss_layer.o


ODIR=./obj
OBJS = $(GNN_LAYER_CXXOBJS) $(COMMON_CXXOBJS) $(UTIL_CXXOBJS) $(LAYER_OBJS) $(LAYER_CXXOBJS)
OBJ = $(patsubst %,$(ODIR)/%,$(OBJS))

all: prova $(OBJ)

$(ODIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CFLAGS)

prova: main.cpp $(OBJ)
	$(CXX) -o $@ main.cpp $(CFLAGS) $(OBJ)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o prova