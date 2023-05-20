#pragma once
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <string>
#include <fstream>
#include <ctime>
#include <memory>
#include <cassert>
#include "pipe.hpp"

struct iofile
{
	struct close_file_deleter
	{
		void operator()(std::fstream* file_stream_ptr)
		{
			if (file_stream_ptr)
			{
				file_stream_ptr->close();
			}
		}
	};

	using file_handler = std::unique_ptr<std::fstream, close_file_deleter>;

	static std::string timestring(std::chrono::system_clock::time_point time = std::chrono::system_clock::now())
	{
		auto in_time_t = std::chrono::system_clock::to_time_t(time);

		std::stringstream ss;

		ss << "newfile";

		// ss << std::put_time(std::localtime(&in_time_t), "%Y_%m_%d-%H_%M_%S");

		return ss.str();
	}

	static file_handler open(const char* path, std::ios_base::openmode mode)
	{
		return file_handler{ new std::fstream(path, std::ios_base::binary | mode) };
	}

	static file_handler open(const char* path)
	{
		return open(path, std::ios_base::out);
	}

	static file_handler open()
	{
		auto name = std::string("snapshot-") + timestring() + ".bin";

		return open(name.c_str(), std::ios_base::out);
	}

	template<typename Tpipe>
	static void import_file(Tpipe& pipe, file_handler& file)
	{
		assert(file);
		assert(file->is_open());

		if (file->is_open())
		{
			pipe.load(*file);
		}
	}

	template<typename Tpipe>
	static void export_file(const Tpipe& pipe, file_handler& file)
	{
		assert(file);
		assert(file->is_open());

		if (file->is_open())
		{
			pipe.save(*file);
		}
	}
};