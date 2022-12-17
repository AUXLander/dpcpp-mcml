# https://stackoverflow.com/questions/60237123/is-there-any-method-to-run-perf-under-wsl
# /home/wsl-user/wls-tools/WSL2-Linux-Kernel/tools/perf

./perf record -g -F 97 ./dpcpp-mcml-cuda
./perf report -g -i ./perf.data