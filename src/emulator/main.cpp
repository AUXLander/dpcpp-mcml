#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "../common/matrix.hpp"
#include "../common/iofile.hpp"

#include "mcml.hpp"

iofile fmanager;

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

template<class T>
struct sycl_host_matrix_allocator
{
    sycl::queue& queue;

    sycl_host_matrix_allocator<T>(sycl::queue& queue) :
        queue{ queue }
    {;}

    sycl_host_matrix_allocator<T>(sycl_host_matrix_allocator<T>& other) :
        queue{ other.queue }
    {;}

    T* allocate(size_t size)
    {
        return sycl::malloc_host<float>(size, queue);
    }

    void free(T* data)
    {
        assert(data);

        if (data)
        {
            sycl::free(data, queue);
        }
    }
};

template<class T>
using host_matrix_view = memory_matrix_view<T, sycl_host_matrix_allocator<T>>;

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

        sycl_host_matrix_allocator<float> allocator(q);

        constexpr size_t N_x = 100;
        constexpr size_t N_y = 100;
        constexpr size_t N_z = 100;
        constexpr size_t N_l = 7;

        constexpr size_t work_group_size = 256;
        constexpr size_t num_groups = 256;

        host_matrix_view<float> host_view(N_x, N_y, N_z, N_l, allocator);

        size_t N = host_view.properties().length();

        float* device_data = sycl::malloc_device<float>(N, q);
        float* device_group_data_pool = sycl::malloc_device<float>(N * num_groups, q);

        InputStruct input;

        input.layerspecs = sycl::malloc_shared<LayerStruct>(N_l, q);

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
                        float* data = device_group_data_pool + N * gid;

                        for (int i = 0; i < N; ++i)
                        {
                            data[i] = 0;
                        }

                        g.parallel_for_work_item(
                            [&](sycl::h_item<1> item)
                            {
                                raw_memory_matrix_view<float> view(data, N_x, N_y, N_z, N_l);

                                uint64_t thread_global_id = item.get_global_id();

                                mcg59_t random_generator(42, thread_global_id, work_group_size * num_groups);

                                PhotonStruct photon(random_generator, view, input, input.layerspecs);

                                auto rsp = Rspecular(input.layerspecs);

                                for (size_t j = 0; j < 1000; ++j)
                                {
                                    photon.init(rsp);

                                    do
                                    {
                                        photon.hop_drop_spin();
                                    } 
                                    while (!photon.dead);
                                }
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
                    float* data = device_group_data_pool + N * gid;

                    summ += data[idx];
                }

                device_data[idx] = summ;

            });

        q.wait();

        q.memcpy(host_view.data(), device_data, host_view.size_of_data());

        q.wait();

        matrix_utils::normalize(host_view);

        auto [size_x, size_y, size_z, size_l] = host_view.properties().size();

        if (size_x < 11 && size_y < 11 && size_z < 11)
        {
            for (size_t y = 0; y < size_y; ++y)
            {
                for (size_t x = 0; x < size_x; ++x)
                {
                    float v = 0.0F;

                    for (size_t z = 0; z < size_z; ++z)
                    {
                        for (size_t l = 0; l < size_l; ++l)
                        {
                            v += host_view.at(x, y, z, l);
                        }
                    }

                    std::cout << std::setw(8) << v << ' ';
                }

                std::cout << '\n';
            }
        }

        {
            auto file = fmanager.open();

            fmanager.export_file(host_view, file);
        }

        sycl::free(input.layerspecs, q);
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
