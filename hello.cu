#include <iostream>

int main(int argc, char* argv[])
{
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    std::cout << numDevices << std::endl;
}