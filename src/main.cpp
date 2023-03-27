#include <CL/sycl.hpp>
//#include <CL/sycl/backend/cuda.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "matrix.hpp"
#include "iofile.hpp"

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

template<class T>
struct sycl_host_allocator
{
    sycl::queue& queue;

    sycl_host_allocator<T>(sycl::queue& queue) :
        queue{ queue }
    {;}

    sycl_host_allocator<T>(sycl_host_allocator<T>& other) :
        queue{ other.queue }
    {;}

    T* allocate(size_t size)
    {
        return sycl::malloc_host<T>(size, queue);
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

struct sycl_base_allocator
{
    sycl_base_allocator(sycl::queue& queue) :
        __queue{ queue }
    {;}

    void free(void* data)
    {
        assert(data);

        if (data)
        {
            sycl::free(data, __queue);

            stat_free();
        }
    }

    ~sycl_base_allocator()
    {
        assert(__stat_allocs_count == __stat_fries_count);

        std::cout << "Allocator stats: " << __stat_allocs_count << " / " << __stat_fries_count << std::endl;
    }

protected:
    sycl::queue& __queue;

    void stat_allocate()
    {
        __stat_allocs_count++;
    }

private:
    size_t __stat_allocs_count{ 0 };
    size_t __stat_fries_count{ 0 };

private:
    void stat_free()
    {
        __stat_fries_count++;
    }
};


struct sycl_shared_allocator : public sycl_base_allocator
{
    sycl_shared_allocator(sycl::queue& queue) :
        sycl_base_allocator(queue)
    {;}

    template<class T>
    T* allocate(size_t size)
    {
        auto pointer = sycl::malloc_shared<T>(size, __queue);

        if (pointer)
        {
            stat_allocate();
        }

        return pointer;
    }
};


struct sycl_device_allocator : public sycl_base_allocator
{
    sycl_device_allocator(sycl::queue& queue) :
        sycl_base_allocator(queue)
    {;}

    template<class T>
    T* allocate(size_t size)
    {
        auto pointer = sycl::malloc_device<T>(size, __queue);

        if (pointer)
        {
            stat_allocate();
        }

        return pointer;
    }
};

template<class T>
struct sycl_unique_ptr
{
    template<class TAlloc>
    sycl_unique_ptr(size_t size, TAlloc& allocator) :
        __allocator { allocator }, __ptr{ allocator. template allocate<T>(size) }
    {;}

    T* get()
    {
        return __ptr;
    }

    ~sycl_unique_ptr()
    {
        __allocator.free(__ptr);
    }

private:
    sycl_base_allocator& __allocator;
    T* __ptr;
};


template<class T>
using host_matrix_view = memory_matrix_view<T, sycl_host_allocator<T>>;

bool has_local_memory(const sycl::device &device)
{
    auto is_host = device.is_host();

    if (!is_host)
    {
        return device.get_info<sycl::info::device::local_mem_type>() != sycl::info::local_mem_type::none;
    }

    return is_host;
}

void print_device_info(const sycl::device& device)
{
    std::cout << "Name:                  " << device.get_info<sycl::info::device::name>()            << std::endl;
    std::cout << "Version:               " << device.get_info<sycl::info::device::version>()         << std::endl;
    std::cout << "Vendor:                " << device.get_info<sycl::info::device::vendor>()          << std::endl;
    std::cout << "Driver version:        " << device.get_info<sycl::info::device::driver_version>()  << std::endl;
    std::cout << "                                                                                 " << std::endl;
    std::cout << "Global memory size:    " << device.get_info<sycl::info::device::global_mem_size>() << std::endl;

    bool has_local_mem = has_local_memory(device);

    if (has_local_mem)
    {
        auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

        std::cout << "Local memory size:     " << local_mem_size << std::endl;
    }
    else
    {
        std::cout << "Device has no avaible local memory" << std::endl;
    }

    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    // Выбор вычислительного устройства
    sycl::gpu_selector d_selector;
    // sycl::cpu_selector d_selector;

    // Вывод характеристик вычислительного устройства
    print_device_info(d_selector.select_device());

    // Параметры записи результатов
    constexpr size_t N_x = 64; //64;
    constexpr size_t N_y = 64; //64;
    constexpr size_t N_z = 64; //64;
    constexpr size_t N_l = 1; // = 12;

    // Параметры симуляции
    constexpr size_t random_seed = 42;
    constexpr size_t number_of_layers = 24;

    // Параметры вычисления
    constexpr size_t N_repeats = 1; //  8'000 / 2; // 0.25 * 1000 / 10;// 8 * 1000 * 2 * 2 * 2; //  8 * 1000;
    constexpr size_t work_group_size = 256; // 32;
    constexpr size_t num_groups = 128 * 2;
    constexpr size_t total_threads_count = num_groups * work_group_size;
    constexpr size_t total_photons_runs = N_repeats * work_group_size * num_groups;

    try 
    {
        sycl::queue queue(d_selector, exception_handler);

        sycl_host_allocator<float> allocator(queue);
        sycl_shared_allocator shared_allocator(queue);
        sycl_device_allocator device_allocator(queue);

        host_matrix_view<float> host_view(N_x, N_y, N_z, N_l, allocator);

        size_t N = host_view.properties().length();

        sycl_unique_ptr<float> data(N, device_allocator);
        sycl_unique_ptr<float> group_data_pool(N * num_groups, device_allocator);
        sycl_unique_ptr<LayerStruct> layerspecs(number_of_layers, shared_allocator);

        InputStruct input;

        input.configure(number_of_layers, N_l, layerspecs.get());

        // Вывод параметров записи результатов
        std::cout << "Image dimensions of x: " << N_x                 << std::endl;
        std::cout << "Image dimensions of y: " << N_y                 << std::endl;
        std::cout << "Image dimensions of z: " << N_z                 << std::endl;
        std::cout << "Image count of layers: " << N_l                 << std::endl;
        std::cout << "                                              " << std::endl;

        // Вывод параметров симуляции
        std::cout << "Random seed:           " << random_seed         << std::endl;
        std::cout << "Inner layes:           " << number_of_layers    << std::endl;
        std::cout << "                                              " << std::endl;

        // Вывод параметров вычисления
        std::cout << "Repeats per tread:     " << N_repeats           << std::endl;
        std::cout << "Work group size:       " << work_group_size     << std::endl;
        std::cout << "Number of groups:      " << num_groups          << std::endl;
        std::cout << "                                              " << std::endl;
        std::cout << "Total treads count:    " << total_threads_count << std::endl;
        std::cout << "Total photon runs:     " << total_photons_runs  << std::endl;
        std::cout << "Total memory used:     " << (N * num_groups + N) * sizeof(float) << " bytes" << std::endl;
        std::cout << "                                              " << std::endl;

        float* device_data = data.get();
        float* device_group_data_pool = group_data_pool.get();

        // Инициализация матрицы значений
        queue.parallel_for(num_groups,
            [=](auto group_index)
            {
                float* data = device_group_data_pool + N * group_index;
                
                for (int i = 0; i < N; ++i)
                {
                    data[i] = 0;
                }
            });

        auto time_start = std::chrono::high_resolution_clock::now();

        // Вычисление задачи на устройстве
        queue.submit(
            [&](sycl::handler& handler)
            {
                //sycl::stream output(1024, 256, handler);

                handler.parallel_for_work_group<class PhotonKernel>(sycl::range<1>(num_groups), sycl::range<1>(work_group_size),
                    [=](sycl::group<1> group) 
                    {
                        size_t gid = group.get_group_id(0); // work group index
                        float* data = device_group_data_pool + N * gid;

#ifdef USE_LOCAL_MEMORY
                        float local_mem[N_x * N_y * N_z * N_l]{ 0.0 };
#endif // USE_LOCAL_MEMORY

                        group.parallel_for_work_item(
                            [&](sycl::h_item<1> item)
                            {
                                uint64_t thread_global_id = item.get_global_id();
                                mcg59_t random_generator(random_seed, thread_global_id, work_group_size * num_groups);

#ifdef USE_LOCAL_MEMORY
                                raw_memory_matrix_view<float> view(local_mem, N_x, N_y, N_z, N_l);
#else
                                raw_memory_matrix_view<float> view(data, N_x, N_y, N_z, N_l);
#endif // USE_LOCAL_MEMORY

                                PhotonStruct photon(random_generator, view, input);

                                auto rsp = Rspecular(input.layerspecs);

                                for (size_t j = 0; j < N_repeats; ++j)
                                {
                                    photon.init(rsp);

                                    do
                                    {
                                        photon.hop_drop_spin();
                                    }
                                    while (!photon.dead);
                                }
                            });
#ifdef USE_LOCAL_MEMORY
                        group.parallel_for_work_item(
                            [&](sycl::h_item<1> item)
                            {
                                constexpr auto batch_size = N_x * N_y * N_z * N_l / work_group_size;

                                auto thread_local_id = item.get_local_id();
                                auto thread_local_offset = batch_size * thread_local_id;

                                for (int i = 0; i < batch_size; ++i)
                                {
                                    data[thread_local_offset + i] = local_mem[thread_local_offset + i];
                                }
                            });
#endif // USE_LOCAL_MEMORY
                    });
            });

        queue.wait();

#if defined(USE_GROUP_SUMMATOR)
        queue.submit(
            [&](sycl::handler& handler)
            {
                constexpr auto work_group_size = N_y;
                constexpr auto work_group_count = N_x;

                static_assert(work_group_count <= 256);
                static_assert(work_group_size <= 256);

                handler.parallel_for_work_group(sycl::range<1>(work_group_count), sycl::range<1>(work_group_size),
                    [=](sycl::group<1> group)
                    {
                        constexpr auto batch_size = (N_x * N_y * N_z * N_l) / (work_group_size * work_group_count);
                        constexpr auto work_group_data_size = work_group_size * batch_size;

                        auto work_group_index = static_cast<size_t>(group.get_group_id());
                        auto work_group_data_offset = work_group_index * work_group_data_size;

                        group.parallel_for_work_item(
                            [&](sycl::h_item<1> item)
                            {
                                auto item_local_index = static_cast<size_t>(item.get_local_id());
                                auto item_local_offset = batch_size * item_local_index;

                                
                                float thread_local_summ_batch[batch_size]{ 0.0 };

                                for (size_t data_group_index = 0; data_group_index < num_groups; ++data_group_index)
                                {
                                    auto device_group_data = device_group_data_pool + N * data_group_index;

                                    auto group_batch_data = device_group_data + work_group_data_offset + item_local_offset;

                                    for (int i = 0; i < batch_size; ++i)
                                    {
                                        thread_local_summ_batch[i] += group_batch_data[i];
                                    }
                                }

                                auto device_batch_data = device_data + work_group_data_offset + item_local_offset;

                                for (int i = 0; i < batch_size; ++i)
                                {
                                    device_batch_data[i] = thread_local_summ_batch[i];
                                }
                            });
                    });
            });
#else
        queue.parallel_for(N,
            [=](auto idx)
            {
                float summ = 0;

                for (size_t data_group_index = 0; data_group_index < num_groups; ++data_group_index)
                {
                    float* data = device_group_data_pool + N * data_group_index;

                    summ += data[idx];
                }

                device_data[idx] = summ;
            });
#endif // !USE_GROUP_SUMMATOR

        queue.wait();

        auto time_end = std::chrono::high_resolution_clock::now();
        auto time_duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);

        std::cout << "Elapsed time: " << time_duration.count() << " ms" << std::endl;

        queue.memcpy(host_view.data(), device_data, host_view.size_of_data());

        queue.wait();

        // 
        matrix_utils::normalize_v2(host_view);

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
    }
    catch (const sycl::exception& e)
    {
        std::cout << "An exception is caught: " << e.what() << std::endl;
        std::terminate();
    }

    return 0;
}
