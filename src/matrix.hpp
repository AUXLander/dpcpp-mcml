#pragma once
#include <cassert>
#include <iostream>

using atomic_array_ref = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::ext_intel_global_device_space>;

class shift_2d
{
	size_t __size_x;
	size_t __size_y;

public:
	constexpr shift_2d(size_t width, size_t height) :
		__size_x(width), __size_y(height)
	{;}

	constexpr size_t index(size_t x, size_t y) const noexcept
	{
		x %= __size_x;
		y %= __size_y;

		return (__size_y * y) + (x);
	}
};

class shift_3d
{
	size_t __size_x;
	size_t __size_y;
	size_t __size_z;

public:
	constexpr shift_3d(size_t width, size_t height, size_t depth) :
		__size_x(width), __size_y(height), __size_z(depth)
	{;}

	constexpr size_t index(size_t x, size_t y, size_t z) const noexcept
	{
		x %= __size_x;
		y %= __size_y;
		z %= __size_z;

		return (__size_y * __size_x * z) + (__size_y * y) + (x);
	}
};

class shift_4d
{
	size_t __size_x;
	size_t __size_y;
	size_t __size_z;
	size_t __size_l;

public:
	constexpr shift_4d(size_t width, size_t height, size_t depth, size_t layer) :
		__size_x(width), __size_y(height), __size_z(depth), __size_l(layer)
	{;}

	constexpr size_t index(size_t x, size_t y, size_t z, size_t l) const noexcept
	{
		x %= __size_x;
		y %= __size_y;
		z %= __size_z;
		l %= __size_l;

		return (__size_y * __size_x * __size_z) * l + (__size_y * __size_x * z) + (__size_y * y) + (x);
	}

	constexpr size_t size_x() const
	{
		return __size_x;
	}

	constexpr size_t size_y() const
	{
		return __size_y;
	}

	constexpr size_t size_z() const
	{
		return __size_z;
	}
};

template<class T>
class raw_memory_matrix_view : public shift_4d
{
	T* __data {nullptr};

public:
	raw_memory_matrix_view(T* data, size_t width, size_t height, size_t depth, size_t count_of_layers) :
		shift_4d(width, height, depth, count_of_layers), __data(data)
	{
		assert(__data);
	}

	T& at(size_t x, size_t y, size_t z, size_t l)
	{
		return *(__data + index(x, y, z, l));
	}

	T& at(size_t x, size_t y, size_t z, size_t l) const
	{
		return *(__data + index(x, y, z, l));
	}

	void save(std::ostream& outstream) const
	{
		for (size_t l = 0; l < size_x(); ++l)
		{
			for (size_t z = 0; z < size_x(); ++z)
			{
				for (size_t y = 0; y < size_x(); ++y)
				{
					for (size_t x = 0; x < size_x(); ++x)
					{
						const auto *ptr = reinterpret_cast<const char*>(&at(x, y, z, l));

						outstream.write(ptr, sizeof(T));
					}
				}
			}
		}
	}

	void load(std::istream& instream)
	{
		for (size_t l = 0; l < size_x(); ++l)
		{
			for (size_t z = 0; z < size_x(); ++z)
			{
				for (size_t y = 0; y < size_x(); ++y)
				{
					for (size_t x = 0; x < size_x(); ++x)
					{
						auto ptr = reinterpret_cast<char*>(&at(x, y, z, l));

						instream.read(ptr, sizeof(T));
					}
				}
			}
		}
	}
};

template<class T>
class matrix_view_adaptor
{
	raw_memory_matrix_view<T>& __view;

public:
	constexpr static double x_min = -0.35;
	constexpr static double x_max = +0.35;

	constexpr static double y_min = -0.25;
	constexpr static double y_max = +0.12;

	constexpr static double z_min = -1.0;
	constexpr static double z_max = +1.0;

public:
	matrix_view_adaptor(raw_memory_matrix_view<T>& view) :
		__view(view)
	{;}

	matrix_view_adaptor(matrix_view_adaptor&) = default;

	T& at(double x, double y, double z, size_t l)
	{
		double x_step = (x_max - x_min) / __view.size_x();
		double y_step = (y_max - y_min) / __view.size_y();
		double z_step = (z_max - z_min) / __view.size_z();

		size_t __x = static_cast<size_t>((x - x_min) / x_step);
		size_t __y = static_cast<size_t>((y - y_min) / y_step);
		size_t __z = static_cast<size_t>((z - z_min) / z_step);
		
		return __view.at(__x, __y,__z, l);
	}
};
