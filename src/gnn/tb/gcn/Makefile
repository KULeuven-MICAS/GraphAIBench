XLEN ?= 32

DEBUG ?= 0
VORTEX ?= 0
USE_GPU ?= 0

LLVM_PREFIX ?= /opt/llvm-riscv
RISCV_TOOLCHAIN_PATH ?= /opt/riscv-gnu-toolchain
SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf
DMP = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump

POCL_CC_PATH ?= /opt/pocl/compiler
POCL_RT_PATH ?= /opt/pocl/runtime

#modified below to avoid shell override
VORTEX_DRV_PATH ?= $(realpath ../../../../../../../driver)# used only for run recipe
VORTEX_RT_PATH ?= $(realpath ../../../../../../../runtime)# used only for kernel compilation

POCL_KERNEL_LLCFLAGS += "-O3 -march=riscv32 -target-abi=ilp32f -mcpu=generic-rv32 -mattr=+m,+f -mattr=+vortex -float-abi=hard -code-model=small"
POCL_KERNEL_CFLAGS   += "-v -O3 -Wstack-usage=1024 --sysroot=$(SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH) -march=rv32imf -mabi=ilp32f -Xclang -target-feature -Xclang +vortex -I$(VORTEX_RT_PATH)/include -fno-rtti -fno-exceptions -ffreestanding -nostartfiles -fdata-sections -ffunction-sections"
POCL_KERNEL_LDFLAGS  += "-Wl,-Bstatic,-T$(VORTEX_RT_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections $(VORTEX_RT_PATH)/libvortexrt.a -lm"

VORTEX_LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_DRV_PATH)/stub -lOpenCL -lvortex
VORTEX_CXXFLAGS += -I$(POCL_RT_PATH)/include

CXX=g++
NVCC=nvcc
IDIR=../../../../include/gnn
INCLUDES=-I../../../../include/utils -I../../../../include/gnn
CFLAGS=$(INCLUDES) -fopenmp -pthread -Wall --std=c++11 -lboost_thread -lboost_system -DERRMSG #-fsanitize=address 

VPATH += ../../cl_utils/
VPATH += ../../

CFLAGS += $(VORTEX_LDFLAGS)
CFLAGS += $(VORTEX_CXXFLAGS)

ODIR=./obj
OBJS = cl_utils.o reader.o lgraph.cl.o cl_math_functions.o

OBJ = $(patsubst %,$(ODIR)/%,$(OBJS))

BIN_DIR=./bin
KBIN_DIR=./bin/kernels

KSRC_DIR=../../cl_cernels

all: prova aggr.pocl

$(ODIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: %.cl
	@mkdir -p $(@D)
	$(CXX) -c -o $@ $< $(CFLAGS)

aggr.pocl: ../cl_kernels/aggr.cl
	@mkdir -p $(KBIN_DIR)
	LLVM_PREFIX=$(LLVM_PREFIX) POCL_DEBUG=all LD_LIBRARY_PATH=$(LLVM_PREFIX)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -LLCFLAGS $(POCL_KERNEL_LLCFLAGS) -CFLAGS $(POCL_KERNEL_CFLAGS) -LDFLAGS $(POCL_KERNEL_LDFLAGS) -o $(KBIN_DIR)/$@ $<

prova: main.cpp $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) -o $(BIN_DIR)/$@ $^ $(CFLAGS)

run: prova aggr.pocl 
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_DRV_PATH)/simx:$(LD_LIBRARY_PATH) $(BIN_DIR)/prova

.PHONY: clean
clean:
	rm -rf $(ODIR)/*.o  $(BIN_DIR)/*