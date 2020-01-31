git pull;
g++ -std=c++11 -o nn_cpu nn.cpp;
nvcc -std=c++11 -o nn_gpu nn.cu;
echo CPU implementation:;
./nn_cpu;
echo GPU implementation:;
./nn_gpu;