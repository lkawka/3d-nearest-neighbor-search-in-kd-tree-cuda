git pull;
g++ -std=c++11 -o nn_cpu nn.cpp;
nvcc -std=c++11 -o nn_gpu nn.cu;
./nn_cpu;
./nn_gpu;