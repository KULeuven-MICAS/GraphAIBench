#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#ifndef BINARY_DIR
#define BINARY_DIR "./bin"
#endif

int main(int argc, char *argv[]) {
    std::cout << "This is a try for -D string define at compile time" << std::endl;
    std::cout << "BIN_DIR is: " << std::string(BINARY_DIR) << std::endl;
    return 0;
}