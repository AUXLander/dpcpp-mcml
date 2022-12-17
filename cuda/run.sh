
#build cuda
$DPCPP_HOME/llvm/build/bin/clang++ -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda main.cpp -o dpcpp-mcml-cuda

# mesure performance
{ time ./dpcpp-mcml-cuda >> performance-gpu-1.txt ; } 2>> performance-gpu-1.txt