source ~/sycl/exports.sh
$DPCPP_HOME/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda main.cpp -o dpcpp-mcml-cuda
