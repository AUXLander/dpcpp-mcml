source ~/sycl/exports.sh
# $DPCPP_HOME/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 ../src/main.cpp -o dpcpp-mcml-cuda

$DPCPP_HOME/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -cl-finite-math-only -cl-no-signed-zeros -cl-fp32-correctly-rounded-divide-sqrt -cl-fast-relaxed-math -cl-unsafe-math-optimizations -ffast-math -O2 ../src/main.cpp -o dpcpp-mcml-cuda

# $DPCPP_HOME/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O2 ../src/main.cpp -o dpcpp-mcml-cuda