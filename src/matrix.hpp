#pragma once
#include <cassert>
#include <cmath>
#include <iostream>
#include <tuple>
#include "pipe.hpp"

#undef __allocator

class matrix_properties
{
	size_t __size_x;
	size_t __size_y;
	size_t __size_z;
	size_t __size_l;

	size_t __size_x_offset;
	size_t __size_y_offset;
	size_t __size_z_offset;
	size_t __size_l_offset;

	size_t __size_limit;

public:
	constexpr matrix_properties(size_t width, size_t height, size_t depth, size_t layer) :
		__size_x(width), __size_y(height), __size_z(depth), __size_l(layer),
		__size_x_offset { 1U }, 
		__size_y_offset { __size_x }, 
		__size_z_offset{ __size_x * __size_y }, 
		__size_l_offset { __size_x * __size_y * __size_z }, 
		__size_limit { __size_x * __size_y * __size_z * __size_l }
	{;}

	constexpr matrix_properties(const matrix_properties&) = default;

	inline size_t index(size_t x, size_t y, size_t z, size_t l) const noexcept
	{
#ifdef  OPT_MTX

		x *= __size_x_offset;
		y *= __size_y_offset;
		z *= __size_z_offset;
		l *= __size_l_offset;

		return x + y + z + l;

#else

		size_t lspec, index;

		lspec = 1U;
		index = lspec * (x % __size_x);

		lspec *= __size_x;
		index += lspec * (y % __size_y);

		lspec *= __size_y;
		index += lspec * (z % __size_z);

		lspec *= __size_z;
		index += lspec * (l % __size_l);

		return index;

#endif //  OPT_MTX
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

	constexpr auto size() const noexcept
	{
		return std::tuple{ __size_x, __size_y, __size_z, __size_l };
	}

	constexpr size_t length() const noexcept
	{
		return __size_limit;
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
struct raw_memory_matrix_view
{
	matrix_properties __props;

protected:
	T*                __data{ nullptr };

protected:
	raw_memory_matrix_view() :
		__props{ 0, 0, 0, 0 }, __data{ nullptr }
	{;}

	raw_memory_matrix_view(raw_memory_matrix_view<T>& other) :
		__props { other.__props }, __data { other.__data }
	{;}

	raw_memory_matrix_view(size_t width, size_t height, size_t depth, size_t count_of_layers) :
		__props{ width, height, depth, count_of_layers }, __data{ nullptr }
	{;}

	void load_props(std::istream& istream)
	{
		__props.load(istream);
	}

	void load_data(std::istream& istream)
	{
		for_each(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				pipe_utils::load_value(istream, at(x, y, z, l));
			}
		);
	}

	void save_props(std::ostream& ostream) const
	{
		__props.save(ostream);
	}

	void save_data(std::ostream& ostream) const
	{
		for_each(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				pipe_utils::save_value(ostream, at(x, y, z, l));
			}
		);
	}

public:
	raw_memory_matrix_view(T* data, size_t width, size_t height, size_t depth, size_t count_of_layers) :
		__props{ width, height, depth, count_of_layers }, __data{ data }
	{
		assert(__data);
	}

	template<class Tlambda>
	void for_each(Tlambda&& for_cell) const
	{
		auto [size_x, size_y, size_z, size_l] = __props.size();

		for (size_t l = 0; l < size_l; ++l)
		{
			for (size_t z = 0; z < size_z; ++z)
			{
				for (size_t y = 0; y < size_y; ++y)
				{
					for (size_t x = 0; x < size_x; ++x)
					{
						for_cell(x, y, z, l);
					}
				}
			}
		}
	}

	inline T* data()
	{
		return __data;
	}

	inline T& at(size_t x, size_t y, size_t z, size_t l)
	{
		return __data[__props.index(x, y, z, l)];
	}

	inline T& at(size_t x, size_t y, size_t z, size_t l) const
	{
		return __data[__props.index(x, y, z, l)];
	}

	void save(std::ostream& ostream) const
	{
		save_props(ostream);
		save_data(ostream);
	}

	const matrix_properties& properties() const
	{
		return __props;
	}

	const size_t size_of_data() const noexcept
	{
		return __props.length() * sizeof(T);
	}
};

template<class T>
struct default_matrix_allocator
{
	default_matrix_allocator<T>(default_matrix_allocator<T>&) = default;

	T* allocate(size_t size)
	{
		return new T[size]{ static_cast<T>(0) };
	}

	void free(T* data)
	{
		assert(data);

		if (data)
		{
			delete[] data;
		}
	}
};

template<class T, class Tallocator = default_matrix_allocator<T>>
class memory_matrix_view : public raw_memory_matrix_view<T>
{
	template<class Talloc>
	class deleter
	{
		Talloc& __allocator;
	public:
		deleter(Talloc& allocator) :
			__allocator{ allocator }
		{;}

		void operator()(T* data)
		{
			__allocator.free(data);
		}
	};

	Tallocator		                         __allocator;
	deleter<Tallocator>				         __deleter;
	size_t                                   __capacity;
	std::unique_ptr<T, deleter<Tallocator>&> __data_owner;

	void realloc()
	{
		__capacity = raw_memory_matrix_view<T>::__props.length();
		__data_owner.reset(__allocator.allocate(__capacity));
		raw_memory_matrix_view<T>::__data = __data_owner.get();
	}

public:
	memory_matrix_view() :
		raw_memory_matrix_view<T>(),
		__allocator{ },
		__deleter{ __allocator },
		__capacity{ 0U },
		__data_owner{ nullptr, __deleter }
	{;}

	memory_matrix_view(Tallocator& allocator) :
		raw_memory_matrix_view<T>(),
		__allocator{ allocator }, 
		__deleter{ __allocator },
		__capacity{ 0U },
		__data_owner { nullptr, __deleter }
	{;}

	memory_matrix_view(size_t width, size_t height, size_t depth, size_t count_of_layers, Tallocator& allocator = Tallocator()) :
		raw_memory_matrix_view<T>(width, height, depth, count_of_layers), 
		__allocator { allocator }, __deleter { __allocator }, __capacity{0U},
		__data_owner{ nullptr, __deleter }
	{
		realloc();
	}

	void save(std::ostream& ostream) const
	{
		raw_memory_matrix_view<T>::save_props(ostream);
		raw_memory_matrix_view<T>::save_data(ostream);
	}

	void load(std::istream& istream)
	{
		raw_memory_matrix_view<T>::load_props(istream);

		if (raw_memory_matrix_view<T>::properties().length() > __capacity)
		{
			realloc();
		}

		raw_memory_matrix_view<T>::load_data(istream);
	}
};

template<class T>
class matrix_view_adaptor
{
	raw_memory_matrix_view<T>& __view;

	double x_step = 0;
	double y_step = 0;
	double z_step = 0;

public:
	                                 
	constexpr static T x_min = -0.25; // -1;
	constexpr static T x_max = +0.25; // +1;
	constexpr static T y_min = -0.25; // -1;
	constexpr static T y_max = +0.25; // +1;
	constexpr static T z_min = 0;
	constexpr static T z_max = +0.25;

	constexpr static double x_length = (x_max - x_min);
	constexpr static double y_length = (y_max - y_min);
	constexpr static double z_length = (z_max - z_min);

public:
	matrix_view_adaptor(raw_memory_matrix_view<T>& view) :
		__view(view)
	{
		auto [size_x, size_y, size_z, size_l] = __view.properties().size();

		x_step = x_length / static_cast<double>(size_x);
		y_step = y_length / static_cast<double>(size_y);
		z_step = z_length / static_cast<double>(size_z);
	}

	matrix_view_adaptor(matrix_view_adaptor<T>&) = default;
	matrix_view_adaptor(matrix_view_adaptor<T>&&) noexcept = default;

	inline T& at(T x, T y, T z, size_t l) noexcept
	{
		static_assert(LAYER_OUTPUT_SIZE_X == 16 || LAYER_OUTPUT_SIZE_X == 32 || LAYER_OUTPUT_SIZE_X == 64 || LAYER_OUTPUT_SIZE_X == 128);

#ifdef OPT_MTX

		size_t __x = (x - x_min) / x_step;
		size_t __y = (y - y_min) / y_step;
		size_t __z = (z - z_min) / z_step;

		//if (__x > LAYER_OUTPUT_SIZE_X - 1 || __y > LAYER_OUTPUT_SIZE_Y - 1 || __z > LAYER_OUTPUT_SIZE_Z - 1)
		//{
		//	__x = 0;
		//	__y = 0;
		//	__z = 0;
		//}
#else

		size_t __x = static_cast<size_t>(libset::clamp<double>(x - x_min, 0.0F, x_length) / x_step);
		size_t __y = static_cast<size_t>(libset::clamp<double>(y - y_min, 0.0F, y_length) / y_step);
		size_t __z = static_cast<size_t>(libset::clamp<double>(z - z_min, 0.0F, z_length) / z_step);

#endif

		__x	&= LAYER_OUTPUT_SIZE_X - 1;
		__y	&= LAYER_OUTPUT_SIZE_Y - 1;
		__z	&= LAYER_OUTPUT_SIZE_Z - 1;

		return __view.at(__x, __y, __z, l);
	}
};

struct matrix_utils
{
	template<class T>
	static void normalize(raw_memory_matrix_view<T>& view)
	{
		T min = view.at(0, 0, 0, 0);
		T max = view.at(0, 0, 0, 0);

		view.for_each(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				auto value = view.at(x, y, z, l);

				min = std::min(min, value);
				max = std::max(max, value);
			});

		view.for_each(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				auto& value = view.at(x, y, z, l);

				value = (value - min) / (max - min);
			});
	}

	template<class T>
	static void normalize_v2(raw_memory_matrix_view<T>& view)
	{
		T min = view.at(0, 0, 0, 0);
		T max = view.at(0, 0, 0, 0);
		size_t zmin = 0;
		size_t zmax = 0;

		view.for_each(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				max = std::max(max, view.at(x, y, z, l));
			});

		T log = log10f(max);

		view.for_each(
			[&](size_t x, size_t y, size_t z, size_t l)
			{
				auto& value = view.at(x, y, z, l);

				value += 1.0;

				min = std::min(min, value);

				value = log10f(value) / log;

				if (value > 0.1)
				{
					zmin = std::min(zmin, z);
					zmax = std::max(zmax, z);
				}

			});
	}
};
