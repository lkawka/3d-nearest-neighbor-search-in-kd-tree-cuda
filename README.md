# 3d nearest neighbor search in kd-tree with CUDA

The goal of a project is to compare the performance of CPU ([nn.cpp](nn.cpp)) and CUDA GPU ([nn.cu](nn.cu)) implementations of the same problem. 
Both solutions find the nearest neighbor for one million query points in the kd-tree consisting of 10000 3 dimensional points. 

To compile and run both implementations, use the [run.sh](run.sh) script.
