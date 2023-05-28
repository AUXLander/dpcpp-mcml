#pragma once
#include <cmath>
#include <random>
#include <CL/sycl.hpp>
#include "configs/default.h"
#include "matrix.hpp"
#include "iofile.hpp"

using atomic_array_ref = sycl::atomic_ref<data_type_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::ext_intel_global_device_space>;

// #define PARTIALREFLECTION 0
  /* 1=split photon, 0=statistical reflection. */

constexpr double COSZERO = 1.0 - 1.0E-12;
/* cosine of about 1e-6 rad. */

constexpr double COS90D = 1.0E-6;
/* cosine of about 1.57 - 1e-6 rad. */

constexpr double PI = 3.1415926F;
constexpr double WEIGHT = 1E-4;		/* Critical weight for roulette. */
constexpr double CHANCE = 0.1; /* Chance of roulette survival. */

constexpr double  MIN_DISTANCE = 1.0E-8;
constexpr double  MIN_WEIGHT = 1.0E-8;

template <typename T>
inline T sgn(T val)
{
	return (T(0) < val) - (val < T(0));
}

template<typename T, typename Y>
inline T setsign(T value, bool sign)
{
	// true is positive

	constexpr auto signbit_offs = sizeof(T) * 8U - 1U;
	constexpr auto signbit_mask = ~(static_cast<Y>(1U) << signbit_offs);

	Y& reinterpret = *reinterpret_cast<Y*>(&value);

	reinterpret = (reinterpret & signbit_mask) | (static_cast<Y>(!sign) << signbit_offs);

	return value;
}

struct mcg59_t
{
	constexpr static uint64_t MCG59_C = 302875106592253;
	constexpr static uint64_t MCG59_M = 576460752303423488;
	constexpr static uint64_t MCG59_DEC_M = 576460752303423487;

	uint64_t value;
	uint64_t offset;

	mcg59_t(uint64_t seed, unsigned int id, unsigned int step)
	{
		uint64_t value = 2 * seed + 1;
		uint64_t firstOffset = RaiseToPower(MCG59_C, id);
		value = (value * firstOffset) & MCG59_DEC_M;

		this->value = value;
		this->offset = RaiseToPower(MCG59_C, step);
	}

	inline data_type_t next()
	{
		this->value = (this->value * this->offset) & MCG59_DEC_M;

		return static_cast<data_type_t>(this->value) / MCG59_M;
	}

	uint64_t RaiseToPower(uint64_t argument, unsigned int power)
	{
		uint64_t result = 1;

		while (power > 0)
		{
			if ((power & 1) == 0)
			{
				argument *= argument;
				power >>= 1;
			}
			else
			{
				result *= argument;
				--power;
			}
		}

		return result;
	}
};

struct LayerStruct
{
	data_type_t z0, z1;	/* z coordinates of a layer. [cm] */
	data_type_t n;			/* refractive index of a layer. */
	data_type_t mua;	    /* absorption coefficient. [1/cm] */
	data_type_t mus;	    /* scattering coefficient. [1/cm] */
	data_type_t anisotropy;		    /* anisotropy. */

	data_type_t cos_crit0, cos_crit1;

	inline bool is_glass() const
	{
		return mua == 0.0 && mus == 0.0;
	}

	LayerStruct() = default;

	LayerStruct(const LayerStruct& o) :
		z0(o.z0),
		z1(o.z1),
		n(o.n),
		mua(o.mua),
		mus(o.mus),
		anisotropy(o.anisotropy),
		cos_crit0(o.cos_crit0),
		cos_crit1(o.cos_crit1)
	{;}
};


struct InputStruct
{
	char out_fname[256];	/* output file name. */
	char out_fformat;		/* output file format. */
	/* 'A' for ASCII, */
	/* 'B' for binary. */
	long num_photons; 		/* to be traced. */
	data_type_t Wth; 				/* play roulette if photon */
	/* weight < Wth.*/

	data_type_t dz;				/* z grid separation.[cm] */
	data_type_t dr;				/* r grid separation.[cm] */
	data_type_t da;				/* alpha grid separation. */
	/* [radian] */
	short nz;					/* array range 0..nz-1. */
	short nr;					/* array range 0..nr-1. */
	short na;					/* array range 0..na-1. */

	short	num_layers;			/* number of layers. */
	short	num_output_layers;
	LayerStruct* layerspecs{ nullptr };	/* layer parameters. */


	InputStruct() = default;

	InputStruct(const InputStruct&) = default;

	void configure(short number_of_layers, short number_of_output_layers, LayerStruct* layerspecs_ptr = nullptr)
	{
		InputStruct& input = *this;

		input.num_photons = 102400;
		input.Wth = 0.0001;
		input.dz = 0.01;
		input.dr = 0.02;
		input.da = 0.015707;
		input.nz = 200;
		input.nr = 500;
		input.na = 100;
		input.num_layers = number_of_layers;
		input.num_output_layers = number_of_output_layers;

		if (input.num_layers && layerspecs_ptr)
		{
			input.layerspecs = layerspecs_ptr;

			configure_layers();
		}
	}

	void configure_layers_v1()
	{
		if (num_layers > 0)
		{
			layerspecs[0].z0 = std::numeric_limits<double>::min();
			layerspecs[0].z1 = std::numeric_limits<double>::min();
			layerspecs[0].n = 1.0;
			layerspecs[0].mua = std::numeric_limits<double>::min();
			layerspecs[0].mus = std::numeric_limits<double>::min();
			layerspecs[0].anisotropy = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit0 = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit1 = std::numeric_limits<double>::min();
		}

		if (num_layers > 1)
		{
			layerspecs[1].z0 = 0;
			layerspecs[1].z1 = 0.01;
			layerspecs[1].n = 1.5;
			layerspecs[1].mua = 3.3;
			layerspecs[1].mus = 107;
			layerspecs[1].anisotropy = 0.79;
			layerspecs[1].cos_crit0 = 0.745355;
			layerspecs[1].cos_crit1 = 0.35901;
		}

		if (num_layers > 2)
		{
			layerspecs[2].z0 = 0.01;
			layerspecs[2].z1 = 0.03;
			layerspecs[2].n = 1.4;
			layerspecs[2].mua = 2.7;
			layerspecs[2].mus = 187.0;
			layerspecs[2].anisotropy = 0.82;
			//layerspecs[2].cos_crit0 = 0.0;
			//layerspecs[2].cos_crit1 = 0.0;
			layerspecs[2].cos_crit0 = 0.745355;
			layerspecs[2].cos_crit1 = 0.35901;
		}

		if (num_layers > 3)
		{
			layerspecs[3].z0 = 0.03;
			layerspecs[3].z1 = 0.05;
			layerspecs[3].n = 1.4;
			layerspecs[3].mua = 2.7;
			layerspecs[3].mus = 187.0;
			layerspecs[3].anisotropy = 0.82;
			//layerspecs[3].cos_crit0 = 0.0;
			//layerspecs[3].cos_crit1 = 0.0;
			layerspecs[3].cos_crit0 = 0.745355;
			layerspecs[3].cos_crit1 = 0.35901;
		}

		if (num_layers > 4)
		{
			layerspecs[3].z0 = 0.05;
			layerspecs[3].z1 = 0.14;
			layerspecs[3].n = 1.4;
			layerspecs[3].mua = 2.7;
			layerspecs[3].mus = 187.0;
			layerspecs[3].anisotropy = 0.82;
			//layerspecs[4].cos_crit0 = 0.0;
			//layerspecs[4].cos_crit1 = 0.0;
			layerspecs[4].cos_crit0 = 0.745355;
			layerspecs[4].cos_crit1 = 0.35901;
		}

		constexpr int START_LAYER = 5;

		for (int i = START_LAYER; i < num_layers; ++i)
		{
			layerspecs[i].z0 = 0.14 + 0.15 * (i - START_LAYER + 0);
			layerspecs[i].z1 = 0.14 + 0.15 * (i - START_LAYER + 1);
			layerspecs[i].n = 1.4;
			layerspecs[i].mua = 2.4;
			layerspecs[i].mus = 194;
			layerspecs[i].anisotropy = 0.82;
			//layerspecs[i].cos_crit0 = 0.0;
			layerspecs[i].cos_crit0 = 0.35901;
			layerspecs[i].cos_crit1 = 0.6998542;
		}

		layerspecs[num_layers - 1].z0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].z1 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].n = 1.0;
		layerspecs[num_layers - 1].mua = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].mus = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].anisotropy = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit1 = std::numeric_limits<double>::min();
	}

	void configure_layers_v2()
	{
		if (num_layers > 0)
		{
			layerspecs[0].z0 = std::numeric_limits<double>::min();
			layerspecs[0].z1 = std::numeric_limits<double>::min();
			layerspecs[0].n = 1.0;
			layerspecs[0].mua = std::numeric_limits<double>::min();
			layerspecs[0].mus = std::numeric_limits<double>::min();
			layerspecs[0].anisotropy = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit0 = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit1 = std::numeric_limits<double>::min();
		}

		if (num_layers > 1)
		{
			layerspecs[1].z0 = 0;
			layerspecs[1].z1 = 0.01;
			layerspecs[1].n = 1.5;
			layerspecs[1].mua = 3.3;
			layerspecs[1].mus = 107;
			layerspecs[1].anisotropy = 0.79;
			layerspecs[1].cos_crit0 = 0.745355;
			layerspecs[1].cos_crit1 = 0.35901;
		}

		if (num_layers > 2)
		{
			layerspecs[2].z0 = 0.01;
			layerspecs[2].z1 = 0.03;
			layerspecs[2].n = 1.4;
			layerspecs[2].mua = 2.7;
			layerspecs[2].mus = 187.0;
			layerspecs[2].anisotropy = 0.82;
			layerspecs[2].cos_crit0 = 0.0;
			layerspecs[2].cos_crit1 = 0.0;
		}

		constexpr int START_LAYER = 3;

		for (int i = START_LAYER; i < num_layers; ++i)
		{
			layerspecs[i].z0 = 0.03 + 0.5 * (i - START_LAYER + 0);
			layerspecs[i].z1 = 0.03 + 0.5 * (i - START_LAYER + 1);
			layerspecs[i].n = 1.4;
			layerspecs[i].mua = 2.7;
			layerspecs[i].mus = 187.0;
			layerspecs[i].anisotropy = 0.82;
			layerspecs[i].cos_crit0 = 0.0;
			layerspecs[i].cos_crit1 = 0.0;
		}

		layerspecs[num_layers - 1].z0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].z1 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].n = 1.0;
		layerspecs[num_layers - 1].mua = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].mus = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].anisotropy = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit1 = std::numeric_limits<double>::min();
	}

	void configure_layers_v3()
	{
		if (num_layers > 0)
		{
			layerspecs[0].z0 = std::numeric_limits<double>::min();
			layerspecs[0].z1 = std::numeric_limits<double>::min();
			layerspecs[0].n = 1.0;
			layerspecs[0].mua = std::numeric_limits<double>::min();
			layerspecs[0].mus = std::numeric_limits<double>::min();
			layerspecs[0].anisotropy = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit0 = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit1 = std::numeric_limits<double>::min();
		}

		constexpr int START_LAYER = 1;

		for (int i = START_LAYER; i < num_layers; ++i)
		{
			layerspecs[i].z0 = 0 + 0.05 * (i - START_LAYER + 0);
			layerspecs[i].z1 = 0 + 0.05 * (i - START_LAYER + 1);
			layerspecs[i].n = 1.4;
			layerspecs[i].mua = 2.7;
			layerspecs[i].mus = 187.0;
			layerspecs[i].anisotropy = 0.82;
			layerspecs[i].cos_crit0 = 0.0;
			layerspecs[i].cos_crit1 = 0.0;
		}

		layerspecs[2].z0 = 0;
		layerspecs[2].z1 = 0.01;
		layerspecs[2].n = 1.5;
		layerspecs[2].mua = 3.3;
		layerspecs[2].mus = 107;
		layerspecs[2].anisotropy = 0.79;
		layerspecs[2].cos_crit0 = 0.745355;
		layerspecs[2].cos_crit1 = 0.35901;

		layerspecs[4].z0 = 0;
		layerspecs[4].z1 = 0.01;
		layerspecs[4].n = 1.5;
		layerspecs[4].mua = 3;
		layerspecs[4].mus = 100;
		layerspecs[4].anisotropy = 0.49;
		layerspecs[4].cos_crit0 = 0.445355;
		layerspecs[4].cos_crit1 = 0.15901;

		layerspecs[5].z0 = 0;
		layerspecs[5].z1 = 0.01;
		layerspecs[5].n = 1.4;
		layerspecs[5].mua = 3;
		layerspecs[5].mus = 90;
		layerspecs[5].anisotropy = 0.89;
		layerspecs[5].cos_crit0 = 0.645355;
		layerspecs[5].cos_crit1 = 0.55901;

		layerspecs[num_layers - 1].z0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].z1 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].n = 1.0;
		layerspecs[num_layers - 1].mua = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].mus = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].anisotropy = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit1 = std::numeric_limits<double>::min();
	}

	void configure_layers_v4()
	{
		if (num_layers > 0)
		{
			layerspecs[0].z0 = std::numeric_limits<double>::min();
			layerspecs[0].z1 = std::numeric_limits<double>::min();
			layerspecs[0].n = 1.0;
			layerspecs[0].mua = std::numeric_limits<double>::min();
			layerspecs[0].mus = std::numeric_limits<double>::min();
			layerspecs[0].anisotropy = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit0 = std::numeric_limits<double>::min();
			layerspecs[0].cos_crit1 = std::numeric_limits<double>::min();
		}

		constexpr int START_LAYER = 1;

		for (int i = START_LAYER; i < num_layers; ++i)
		{
			layerspecs[i].z0 = 0 + 0.05 * (i - START_LAYER + 0);
			layerspecs[i].z1 = 0 + 0.05 * (i - START_LAYER + 1);
			layerspecs[i].n = 1.4;
			layerspecs[i].mua = 2.7;
			layerspecs[i].mus = 187.0;
			layerspecs[i].anisotropy = 0.82;
			layerspecs[i].cos_crit0 = 0.0;
			layerspecs[i].cos_crit1 = 0.0;
		}

		layerspecs[num_layers - 1].z0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].z1 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].n = 1.0;
		layerspecs[num_layers - 1].mua = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].mus = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].anisotropy = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit0 = std::numeric_limits<double>::min();
		layerspecs[num_layers - 1].cos_crit1 = std::numeric_limits<double>::min();
	}

	void configure_layers()
	{
		return configure_layers_v4();
	}

	void free()
	{
		if (layerspecs)
		{
			::free(layerspecs);
		}
	}
};

data_type_t RFresnel(data_type_t incidentRefractiveIndex, data_type_t transmitRefractiveIndex, data_type_t incidentCos, data_type_t& transmitCos)
{
	data_type_t reflectance;

	if (incidentRefractiveIndex == transmitRefractiveIndex)
	{
		transmitCos = incidentCos;
		reflectance = 0.0;
	}
	else if (incidentCos > COSZERO)
	{
		transmitCos = incidentCos;
		reflectance = (transmitRefractiveIndex - incidentRefractiveIndex) / (transmitRefractiveIndex + incidentRefractiveIndex);
		reflectance *= reflectance;
	}
	else if (incidentCos < COS90D)
	{
		transmitCos = 0.0;
		reflectance = 1.0;
	}
	else
	{
		data_type_t incidentSin = libset::sqrt(1.0F - incidentCos * incidentCos);
		data_type_t transmitSin = incidentRefractiveIndex * incidentSin / transmitRefractiveIndex;

		if (transmitSin >= 1.0)
		{
			transmitCos = 0.0;
			reflectance = 1.0;
		}
		else
		{
			transmitCos = libset::sqrt(1.0 - transmitSin * transmitSin);

			data_type_t cap = incidentCos * transmitCos - incidentSin * transmitSin; /* c+ = cc - ss. */
			data_type_t cam = incidentCos * transmitCos + incidentSin * transmitSin; /* c- = cc + ss. */
			data_type_t sap = incidentSin * transmitCos + incidentCos * transmitSin; /* s+ = sc + cs. */
			data_type_t sam = incidentSin * transmitCos - incidentCos * transmitSin; /* s- = sc - cs. */

			reflectance = 0.5F * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
		}
	}

	return reflectance;
}

struct PhotonStruct
{
	template<class T>
	class weight_tracker
	{
		T                      __weight;
		PhotonStruct&          __ps;
		matrix_view_adaptor<T> __view;

		inline void __track(T value)
		{
			if constexpr (CONFIGURATION_WORK_GROUP_THREADS_COUNT == 1U)
			{
				__view.at(__ps.x, __ps.y, __ps.z, 0) += value;
			}
			else
			{

#ifdef FEATURE_USE_LOCAL_MEMORY
				sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
					atomic(__view.at(__ps.x, __ps.y, __ps.z, 0));
#else
				sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::ext_intel_global_device_space>
					atomic(__view.at(__ps.x, __ps.y, __ps.z, 0));
#endif

				atomic.fetch_add(value);
			}
		}

	public:

		weight_tracker(data_type_t weight, PhotonStruct& ps, matrix_view_adaptor<T>& view) :
			__weight{ weight }, __ps{ ps }, __view{ view }
		{;}

		weight_tracker<T>& operator=(T value)
		{
			__weight = value;

			return *this;
		}

		weight_tracker<T>& operator+=(T value)
		{
			__track(value);

			return *this;
		}

		weight_tracker<T>& operator-=(T value)
		{
			__track(value);

			__weight -= value;

			return *this;
		}

		weight_tracker<T>& operator/=(T value)
		{
			T tmp = __weight;

			__weight /= value;

			__track(tmp - __weight);

			return *this;
		}

		weight_tracker<T>& operator*=(T value)
		{
			T tmp = __weight;

			__weight *= value;

			__track(tmp - __weight);

			return *this;
		}

		inline T operator*(T value) const
		{
			return __weight * value;
		}

		inline bool operator==(T value) const
		{
			return __weight == value;
		}

		inline bool operator!=(T value) const
		{
			return __weight != value;
		}

		inline bool operator<(T value) const
		{
			return __weight < value;
		}
	};

	data_type_t x{ 0 }, y{ 0 }, z{ 0 };    // vector of position
	data_type_t ux{ 0 }, uy{ 0 }, uz{ 0 }; // vector of direction

	weight_tracker<data_type_t> w;

	bool dead{ false };

	size_t layer{ 0 };

	data_type_t sleft{ 0 };
	data_type_t step_size{ 0 };

	mcg59_t &random;

	const InputStruct& input;

	const LayerStruct* layerspecs;

	PhotonStruct(mcg59_t& random, matrix_view_adaptor<data_type_t> view, const InputStruct& input) :
		w{ 0.0, *this, view }, random{ random }, input{ input }, layerspecs{ input.layerspecs }
	{;}

	~PhotonStruct() = default;

	void init(data_type_t Rspecular)
	{
		w = 1.0 - Rspecular;
		dead = 0;
		layer = 1; // LAYER CHANGE
		step_size = 0;
		sleft = 0;

		x = 0.0; // COORD CHANGE
		y = 0.0;
		z = 0.0;

		ux = 0.0;
		uy = 0.0;
		uz = 1.0;

		if ((layerspecs[1].mua == 0.0) && (layerspecs[1].mus == 0.0))
		{
			layer = 2; // LAYER CHANGE
			z = layerspecs[2].z0;
		}
	}

	void spin(data_type_t anisotropy)
	{
		const auto ux = this->ux;
		const auto uy = this->uy;
		const auto uz = this->uz;

		const auto cost = SpinTheta(anisotropy);
		const auto sint = libset::sqrt(1.0F - cost * cost);

		const auto psi = 2.0F * (data_type_t)PI * get_random();

		const auto cosp = libset::cos(psi);


		data_type_t sinp; // = std::sin(psi);

		if (psi < PI)
		{
			sinp = libset::sqrt(1.0F - cosp * cosp);
		}
		else
		{
			sinp = -libset::sqrt(1.0F - cosp * cosp);
		}

		if (libset::abs<data_type_t>(uz) > COSZERO)
		{
			const auto temp = (uz >= 0.0) ? 1.0F : -1.0F;

			this->ux = sint * cosp;
			this->uy = sint * sinp;
			this->uz = temp * cost;
		}
		else
		{
			const auto temp = sycl::sqrt<data_type_t>(1.0F - uz * uz);

			this->ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost;
			this->uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost;
			this->uz = -sint * cosp * temp + uz * cost;
		}
	}

	void hop()
	{
		// COORD CHANGE

		x += step_size * ux;
		y += step_size * uy;
		z += step_size * uz;
	}

	void step_size_in_glass()
	{
		const auto& olayer = get_current_layer();

		// TODO: solve divergency problem
		// const auto layer_z = (uz > 0.0) ? olayer.z1 : (uz < 0.0) ? olayer.z0 : z;
		// step_size = (layer_z - z) / uz;

		if (uz > 0.0)
		{
			step_size = (olayer.z1 - z) / uz;
		}
		else if (uz < 0.0)
		{
			step_size = (olayer.z0 - z) / uz;
		}
		else
		{
			step_size = 0.0;
		}
	}

	bool hit_boundary()
	{
		if (libset::abs(uz) > 1e-10)
		{
			const auto& olayer = get_current_layer();

			data_type_t dl_b;

			if (uz > 0.0)
			{
				dl_b = olayer.z1;
			}
			else
			{
				dl_b = olayer.z0;
			}

			dl_b = (dl_b - z) / uz;

			if (step_size > dl_b)
			{
				const auto mut = olayer.mua + olayer.mus;

				sleft = (step_size - dl_b) * mut;
				step_size = dl_b;

				return true;
			}
		}
		
		return false;
	}

	void roulette()
	{
		if (w != 0.0F && get_random() < CHANCE)
		{
			w /= CHANCE;
		}
		else
		{
			dead = true;
		}
	}

	void record_r(data_type_t reflectance)
	{
		w *= reflectance;
	}

	void record_t(data_type_t reflectance)
	{
		w *= reflectance;
	}

	void drop()
	{
		const auto& olayer = get_current_layer();

		auto mua = olayer.mua;
		auto mus = olayer.mus;

		auto dwa = w * mua / (mua + mus);

		w -= dwa;
	}

	void cross_up_or_not()
	{
		data_type_t uz1 = 0.0;
		data_type_t r = 0.0;
		data_type_t ni = layerspecs[layer].n;
		data_type_t nt = layerspecs[layer - 1].n;

		if (-uz <= layerspecs[layer].cos_crit0)
		{
			r = 1.0;
		}
		else
		{
			r = RFresnel(ni, nt, -uz, uz1);
		}

		if (get_random() > r)
		{
			if (layer == 1)
			{
				uz = -uz1;

				// Do not record out of layer photons
				// record_r(0.0);
				dead = true;
			}
			else
			{
				layer--; // LAYER CHANGE

				ux *= ni / nt;
				uy *= ni / nt;
				uz = -uz1;
			}
		}
		else
		{
			uz = -uz;
		}
	}

	void cross_down_or_not()
	{
		data_type_t uz1 = 0.0;
		data_type_t r = 0.0;
		data_type_t ni = layerspecs[layer].n;
		data_type_t nt = layerspecs[layer + 1].n;

		if (uz <= layerspecs[layer].cos_crit1)
		{
			r = 1.0;
		}
		else
		{
			r = RFresnel(ni, nt, uz, uz1);
		}

		if (get_random() > r)
		{
			if (layer == input.num_layers)
			{
				uz = uz1;
				record_t(0.0);
				dead = true;
			}
			else
			{
				layer++; // LAYER CHANGE

				ux *= ni / nt;
				uy *= ni / nt;
				uz = uz1;
			}
		}
		else
		{
			uz = -uz;
		}
	}

	void cross_or_not()
	{
		if (uz < 0.0)
		{
			cross_up_or_not();
		}
		else
		{
			cross_down_or_not();
		}
	}

	data_type_t SpinTheta(data_type_t anisotropy)
	{
		data_type_t cost = 2.0 * get_random() - 1.0;

		if (libset::abs(anisotropy) > 1e-10)
		{
			const data_type_t anisotropy_sqr = anisotropy * anisotropy;

			const data_type_t temp = (1.0 - anisotropy_sqr) / (1.0 + anisotropy * cost);

			cost = 0.5 * (1.0 + anisotropy_sqr - temp * temp) / anisotropy;

			cost = libset::clamp<data_type_t>(cost, -1.0f, +1.0f);
		}

		return cost;
	}

	void hop_in_glass()
	{
		if (uz == 0.0)
		{
			dead = true;
		}
		else
		{
			step_size_in_glass();
			hop();
			cross_or_not();
		}
	}

	void hop_drop_spin()
	{
		const auto& olayer = get_current_layer();

		if (olayer.is_glass())
		{
			hop_in_glass();
		}
		else
		{
			const auto mua = olayer.mua;
			const auto mus = olayer.mus;

			data_type_t rnd;

#ifdef OPT_RAND

			do
			{
				rnd = get_random();
			} while (rnd <= 0.0);

			sleft = libset::abs<data_type_t>(sleft) < 1e-10 ? -sycl::log(rnd) : sleft;

			step_size = sleft / (mua + mus);

#else

			if (sycl::abs(sleft) < 1e-10)
			{
				data_type_t rnd;

				do
				{
					rnd = get_random();
				} 
				while (rnd <= 0.0);

				step_size = -sycl::log(rnd) / (mua + mus);
			}
			else
			{
				step_size = sleft / (mua + mus);
			}

#endif
			sleft = 0.0;

			if (hit_boundary())
			{
				hop();
				cross_or_not();
			}
			else
			{
				hop();
				drop();
				spin(olayer.anisotropy);
			}
		}

		if (w < input.Wth && !dead)
		{
			roulette();
		}
	}

	inline const LayerStruct& get_current_layer() const
	{
		return layerspecs[layer];
	}

	inline data_type_t get_random()
	{
		return random.next();
	}
};

data_type_t Rspecular(LayerStruct* Layerspecs_Ptr)
{
	data_type_t r1;
	data_type_t r2;
	data_type_t temp;

	temp = (Layerspecs_Ptr[0].n - Layerspecs_Ptr[1].n) / (Layerspecs_Ptr[0].n + Layerspecs_Ptr[1].n);
	r1 = temp * temp;

	if ((Layerspecs_Ptr[1].mua == 0.0) && (Layerspecs_Ptr[1].mus == 0.0))
	{
		temp = (Layerspecs_Ptr[1].n - Layerspecs_Ptr[2].n) / (Layerspecs_Ptr[1].n + Layerspecs_Ptr[2].n);

		r2 = temp * temp;

		r1 = r1 + (1 - r1) * (1 - r1) * r2 / (1 - r1 * r2);
	}

	return r1;
}
