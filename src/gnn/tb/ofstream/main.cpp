#include <fstream>
#include <iostream>
#include <random>
#include <string>

int main() {
    int size = 1000;
    float A[size];
    float B[size];
    std::string file_path = "test.dat";
    for (int i = 0; i < size; i++) {
        A[i] = rand();
    }
    std::ofstream ofs(file_path, std::ios::out | std::ios::binary);
    //output serialization
    for(int i = 0; i < size; i++) {
        ofs.write((char*)&A[i], sizeof(float));
    }
    ofs.close();

    //input serialization
    std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
    for(int i = 0; i < size; i++) {
        ifs.read((char*)&B[i], sizeof(float));
    }
    ifs.close();

    //test 
    for (int i = 0; i < size; i++) {
        if (A[i] != B[i]) {
            std::cout << "ERROR" << std::endl;
            return 1;
        }
    }
    std::cout << "SUCCESS" << std::endl;
}