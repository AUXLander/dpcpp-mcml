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
	static void save_raw_data(std::ostream& ostream, const T* data, size_t length)
	{
		ostream.write(reinterpret_cast<const char*>(data), length * sizeof(T));
	}

	static void save_string(std::ostream& ostream, const std::string& string)
	{
		size_t length = string.size();

		save_value(ostream, length);
		save_raw_data(ostream, string.c_str(), string.size());
	}



	template<typename T>
	static void load_value(std::istream& istream, T& value)
	{
		istream.read(reinterpret_cast<char*>(&value), sizeof(T));
	}

	template<typename T>
	static void load_raw_data(std::istream& istream, T* data, size_t length)
	{
		istream.read(reinterpret_cast<char*>(data), length * sizeof(T));
	}

	template<typename T>
	static void load_string(std::istream& istream, std::string& string)
	{
		size_t length = 0U;

		load_value(istream, length);

		string.resize(length);

		load_raw_data(istream, string.data(), string.size());
	}
};