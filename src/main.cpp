#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include "matrix.hpp"
#include "mcml.hpp"

static auto exception_handler = [](sycl::exception_list e_list)
{
    for (std::exception_ptr const& e : e_list) 
    {
        try 
        {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) 
        {
            std::cout << "exception: " << e.what() << std::endl;
           
            std::terminate();
        }
    }
};

class cuda_selector : public sycl::device_selector {
public:
    int operator()(const sycl::device& device) const override 
    {
        return device.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda;
        //return device.is_gpu() && (device.get_info<sycl::info::device::driver_version>().find("CUDA") != std::string::npos);
    }
};

int main(int argc, char* argv[])
{
    // cuda_selector d_selector;
    // sycl::cpu_selector d_selector;
    sycl::gpu_selector d_selector;
    auto dev = d_selector.select_device();

    std::cout << "Name: " << dev.get_info<cl::sycl::info::device::name>() << std::endl;
    std::cout << "Version: " << dev.get_info<cl::sycl::info::device::version>() << std::endl;
    std::cout << "Vendor: " << dev.get_info<cl::sycl::info::device::vendor>() << std::endl;
    std::cout << "Driver version: " << dev.get_info<cl::sycl::info::device::driver_version>() << std::endl;

    try 
    {
        sycl::queue q(d_selector, exception_handler);

        constexpr size_t N_x = 10;
        constexpr size_t N_y = 10;
        constexpr size_t N_z = 10;
        constexpr size_t N_l = 7;

        constexpr size_t N = N_x * N_y * N_z * N_l;
        constexpr size_t work_group_size = 256;
        constexpr size_t num_groups = 256;

        int* host_data = sycl::malloc_host<int>(N, q);
        int* device_data = sycl::malloc_device<int>(N, q);
        int* device_group_data_pool = sycl::malloc_device<int>(N * num_groups, q);

        InputStruct input;

        input.layerspecs = sycl::malloc_shared<LayerStruct>(7, q);

        configure_input(input);
        configure(input.layerspecs, q);

        q.submit(
            [&](sycl::handler& h) 
            {
                sycl::stream output(1024, 256, h);

                h.parallel_for_work_group(sycl::range<1>(num_groups), sycl::range<1>(work_group_size),
                    [=](sycl::group<1> g) 
                    {
                        size_t gid = g.get_group_id(0); // work group index

                        // select from allocated memory on device
                        int* data = device_group_data_pool + N * gid;

                        for (int i = 0; i < N; ++i)
                        {
                            data[i] = 0;
                        }

                        g.parallel_for_work_item(
                            [&](sycl::h_item<1> item)
                            {
                                raw_memory_matrix_view<int> view(data, N_x, N_y, N_z, N_l);

                                PhotonStruct photon(view, input, input.layerspecs);

                                auto rsp = Rspecular(input.layerspecs);

                                for (size_t j = 0; j < 1; ++j)
                                {
                                    photon.init(rsp);

                                    do
                                    {
                                        photon.hop_drop_spin();
                                    } 
                                    while (!photon.dead);
                                }

                                // int i = item.get_global_id(0);
                                // int i = item.get_local_id(); // work item index [0...work_group_size]
                                // auto v = atomic_array_ref(view.at(i, gid, 0, 0));
                                // v.fetch_add(1);
                            });
                    });
            });

        q.wait();

        q.parallel_for(N,
            [=](auto idx)
            {
                int summ = 0;

                for (size_t gid = 0; gid < num_groups; ++gid)
                {
                    int* data = device_group_data_pool + N * gid;

                    summ += data[idx];
                }

                device_data[idx] = summ;

            });

        q.wait();

        q.memcpy(host_data, device_data, sizeof(int) * N);

        q.wait();
    
        int index = 0;

        for (int i = 0; i < N_y; ++i)
        {
            for (int j = 0; j < N_x; ++j)
            {
                int v = 0;
                int layer_offset = 0; // N_x* N_y* N_z * 3;

                for (int k = 0; k < N_z; ++k)
                {
                    v += host_data[layer_offset + N_x * N_y * k + index];
                }

                std::cout << std::setw(8) << v << ' ';

                ++index;
            }

            std::cout << '\n';
        }

        sycl::free(input.layerspecs, q);
        sycl::free(host_data, q);
        sycl::free(device_data, q);
        sycl::free(device_group_data_pool, q);
    }
    catch (const sycl::exception& e)
    {
        std::cout << "An exception is caught: " << e.what() << std::endl;
        std::terminate();
    }

    return 0;
}
