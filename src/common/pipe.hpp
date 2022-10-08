#pragma once
#include <iostream>

struct pipe_utils
{
	template<typename T>
	static void save_value(std::ostream& ostream, const T& value)
	{
		ostream.write(reinterpret_cast<const char*>(&value), sizeof(T));
	}

	template<typename T>
	static void load_value(std::istream& istream, T& value)
	{
		istream.read(reinterpret_cast<char*>(&value), sizeof(T));
	}
};