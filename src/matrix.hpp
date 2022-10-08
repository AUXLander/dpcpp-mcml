#pragma once
#include <cassert>
#include <iostream>
#include "pipe.hpp"

using atomic_array_ref = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::ext_intel_global_device_space>;

class matrix_properties
{
	size_t __size_x;
	size_t __size_y;
	size_t __size_z;
	size_t __size_l;

public:
	constexpr matrix_properties(size_t width, size_t height, size_t depth, size_t layer) :
		__size_x(width), __size_y(height), __size_z(depth), __size_l(layer)
	{;}

	constexpr size_t index(size_t x, size_t y, size_t z, size_t l) const noexcept
	{
		size_t lspec, index;
		
		lspec  = 1U;
		index  = lspec * (x % __size_x);

		lspec *= __size_x;
		index += lspec * (y % __size_y);

		lspec *= __size_y;
		index += lspec * (z % __size_z);

		lspec *= __size_z;
		index += lspec * (l % __size_l);

		return index;
	}

	constexpr size_t size_x() const noexcept
	{
		return __size_x;
	}

	constexpr size_t size_y() const noexcept
	{
		return __size_y;
	}

	constexpr size_t size_z() const noexcept
	{
		return __size_z;
	}

	constexpr size_t size_l() const noexcept
	{
		return __size_l;
	}

	void save(std::ostream& ostream) const
	{
		pipe_utils::save_value(ostream, __size_x);
		pipe_utils::save_value(ostream, __size_y);
		pipe_utils::save_value(ostream, __size_z);
		pipe_utils::save_value(ostream, __size_l);
	}

	void load(std::istream& istream)
	{
		pipe_utils::load_value(istream, __size_x);
		pipe_utils::load_value(istream, __size_y);
		pipe_utils::load_value(istream, __size_z);
		pipe_utils::load_value(istream, __size_l);
	}
};

template<class T>
class raw_memory_matrix_view
{
	matrix_properties __props;
	T*                __data {nullptr};

	template<class Tlambda>
	inline void __for_each_cell(Tlambda&& for_cell) const
	{
		for (size_t l = 0; l < __props.size_l(); ++l)
		{
			for (size_t z = 0; z < __props.size_z(); ++z)
			{
				for (size_t y = 0; y < __props.size_y(); ++y)
				{
					for (size_t x = 0; x < __props.size_x(); ++x)
					{
						for_cell(x, y, z, l);
					}
				}
			}
		}
	}

public:

	const matrix_properties& properties;

public:
	raw_memory_matrix_view(T* data, size_t width, size_t height, size_t depth, size_t count_of_layers) :
		__props{ width, height, depth, count_of_layers }, __data{ data }, properties{ __props }
	{
		assert(__data);
	}

	T& at(size_t x, size_t y, size_t z, size_t l)
	{
		return __data[__props.index(x, y, z, l)];
	}

	T& at(size_t x, size_t y, size_t z, size_t l) const
	{
		return __data[__props.index(x, y, z, l)];
	}

	void save_props(std::ostream& ostream) const
	{
		__props.save(ostream);
	}

	void save_data(std::ostream& ostream) const
	{
		__for_each_cell(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				pipe_utils::save_value(ostream, at(x, y, z, l));
			}
		);
	}

	void save(std::ostream& ostream) const
	{
		save_props(ostream);
		save_data(ostream);
	}

	void load_props(std::istream& istream)
	{
		__props.load(istream);
	}

	void load_data(std::istream& istream)
	{
		__for_each_cell(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				pipe_utils::load_value(istream, at(x, y, z, l));
			}
		);
	}

	void load(std::istream& istream)
	{
		load_props(istream);
		load_data(istream);
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

	matrix_view_adaptor(matrix_view_adaptor<T>&) = default;
	matrix_view_adaptor(matrix_view_adaptor<T>&&) noexcept = default;

	T& at(double x, double y, double z, size_t l)
	{
		double x_step = (x_max - x_min) / __view.properties.size_x();
		double y_step = (y_max - y_min) / __view.properties.size_y();
		double z_step = (z_max - z_min) / __view.properties.size_z();

		size_t __x = static_cast<size_t>((x - x_min) / x_step);
		size_t __y = static_cast<size_t>((y - y_min) / y_step);
		size_t __z = static_cast<size_t>((z - z_min) / z_step);
		
		return __view.at(__x, __y,__z, l);
	}
};
