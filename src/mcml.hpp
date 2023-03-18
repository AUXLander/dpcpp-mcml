#pragma once
#include <cmath>
#include <random>
#include <CL/sycl.hpp>
//#include <oneapi/dpl/random>
#include "matrix.hpp"
#include "iofile.hpp"

using atomic_array_ref = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::ext_intel_global_device_space>;

// #define PARTIALREFLECTION 0
  /* 1=split photon, 0=statistical reflection. */

constexpr double COSZERO = 1.0 - 1.0E-12;
/* cosine of about 1e-6 rad. */

constexpr double COS90D = 1.0E-6;
/* cosine of about 1.57 - 1e-6 rad. */

constexpr double PI = 3.1415926;
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

	double next()
	{
		this->value = (this->value * this->offset) & MCG59_DEC_M;

		return (double)(this->value) / MCG59_M;
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
	double z0, z1;	/* z coordinates of a layer. [cm] */
	double n;			/* refractive index of a layer. */
	double mua;	    /* absorption coefficient. [1/cm] */
	double mus;	    /* scattering coefficient. [1/cm] */
	double anisotropy;		    /* anisotropy. */

	double cos_crit0, cos_crit1;

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
	double Wth; 				/* play roulette if photon */
	/* weight < Wth.*/

	double dz;				/* z grid separation.[cm] */
	double dr;				/* r grid separation.[cm] */
	double da;				/* alpha grid separation. */
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

	void configure_layers()
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

		if (num_layers > 3)
		{
			//layerspecs[3].z0 = 0.03;
			//layerspecs[3].z1 = 0.05;
			//layerspecs[3].n = 1.4;
			//layerspecs[3].mua = 3.3;
			//layerspecs[3].mus = 192.0;
			//layerspecs[3].anisotropy = 0.82;
			//layerspecs[3].cos_crit0 = 0.0;
			//layerspecs[3].cos_crit1 = 0.0;

			layerspecs[3].z0 = 0.01;
			layerspecs[3].z1 = 0.03;
			layerspecs[3].n = 1.4;
			layerspecs[3].mua = 2.7;
			layerspecs[3].mus = 187.0;
			layerspecs[3].anisotropy = 0.82;
			layerspecs[3].cos_crit0 = 0.0;
			layerspecs[3].cos_crit1 = 0.0;
		}

		if (num_layers > 4)
		{
			layerspecs[4].z0 = 0.05;
			layerspecs[4].z1 = 0.14;
			layerspecs[4].n = 1.4;
			layerspecs[4].mua = 2.7;
			layerspecs[4].mus = 187.0;
			layerspecs[4].anisotropy = 0.82;
			layerspecs[4].cos_crit0 = 0.0;
			layerspecs[4].cos_crit1 = 0.0;
		}

		if (num_layers > 5)
		{
			layerspecs[5].z0 = 0.14;
			layerspecs[5].z1 = 0.2;
			layerspecs[5].n = 1.4;
			layerspecs[5].mua = 2.4;
			layerspecs[5].mus = 194;
			layerspecs[5].anisotropy = 0.82;
			layerspecs[5].cos_crit0 = 0.0;
			layerspecs[5].cos_crit1 = 0.6998542;
		}

		if (num_layers > 6)
		{
			layerspecs[6].z0 = 0.14;
			layerspecs[6].z1 = 0.2;
			layerspecs[6].n = 1.4;
			layerspecs[6].mua = 2.4;
			layerspecs[6].mus = 194;
			layerspecs[6].anisotropy = 0.82;
			layerspecs[6].cos_crit0 = 0.0;
			layerspecs[6].cos_crit1 = 0.6998542;
		}

		if (num_layers > 7)
		{
			layerspecs[7].z0 = 0.14;
			layerspecs[7].z1 = 0.2;
			layerspecs[7].n = 1.4;
			layerspecs[7].mua = 2.4;
			layerspecs[7].mus = 194;
			layerspecs[7].anisotropy = 0.82;
			layerspecs[7].cos_crit0 = 0.0;
			layerspecs[7].cos_crit1 = 0.6998542;
		}

		if (num_layers > 8)
		{
			layerspecs[8].z0 = 0.14;
			layerspecs[8].z1 = 0.2;
			layerspecs[8].n = 1.4;
			layerspecs[8].mua = 2.4;
			layerspecs[8].mus = 194;
			layerspecs[8].anisotropy = 0.82;
			layerspecs[8].cos_crit0 = 0.0;
			layerspecs[8].cos_crit1 = 0.6998542;
		}

		if (num_layers > 9)
		{
			layerspecs[9].z0 = 0.14;
			layerspecs[9].z1 = 0.2;
			layerspecs[9].n = 1.4;
			layerspecs[9].mua = 2.4;
			layerspecs[9].mus = 194;
			layerspecs[9].anisotropy = 0.82;
			layerspecs[9].cos_crit0 = 0.0;
			layerspecs[9].cos_crit1 = 0.6998542;
		}

		if (num_layers > 10)
		{
			layerspecs[10].z0 = 0.14;
			layerspecs[10].z1 = 0.2;
			layerspecs[10].n = 1.4;
			layerspecs[10].mua = 2.4;
			layerspecs[10].mus = 194;
			layerspecs[10].anisotropy = 0.82;
			layerspecs[10].cos_crit0 = 0.0;
			layerspecs[10].cos_crit1 = 0.6998542;
		}

		if (num_layers > 11)
		{
			layerspecs[11].z0 = std::numeric_limits<double>::min();
			layerspecs[11].z1 = std::numeric_limits<double>::min();
			layerspecs[11].n = 1.0;
			layerspecs[11].mua = std::numeric_limits<double>::min();
			layerspecs[11].mus = std::numeric_limits<double>::min();
			layerspecs[11].anisotropy = std::numeric_limits<double>::min();
			layerspecs[11].cos_crit0 = std::numeric_limits<double>::min();
			layerspecs[11].cos_crit1 = std::numeric_limits<double>::min();
		}
	}

	void free()
	{
		if (layerspecs)
		{
			::free(layerspecs);
		}
	}
};

double RFresnel(double incidentRefractiveIndex, double transmitRefractiveIndex, double incidentCos, double &transmitCos)
{
	double reflectance;

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
		double incidentSin = sycl::sqrt(1.0 - incidentCos * incidentCos);
		double transmitSin = incidentRefractiveIndex * incidentSin / transmitRefractiveIndex;

		if (transmitSin >= 1.0)
		{
			transmitCos = 0.0;
			reflectance = 1.0;
		}
		else
		{
			transmitCos = sycl::sqrt(1.0 - transmitSin * transmitSin);

			double cap = incidentCos * transmitCos - incidentSin * transmitSin; /* c+ = cc - ss. */
			double cam = incidentCos * transmitCos + incidentSin * transmitSin; /* c- = cc + ss. */
			double sap = incidentSin * transmitCos + incidentCos * transmitSin; /* s+ = sc + cs. */
			double sam = incidentSin * transmitCos - incidentCos * transmitSin; /* s- = sc - cs. */

			reflectance = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
		}
	}

	return reflectance;
}

struct PhotonStruct
{
	template<class T>
	class weight_tracker
	{
		double                 __weight;
		PhotonStruct&          __ps;
		matrix_view_adaptor<T> __view;

		inline void __track(float value)
		{
			// __view.at(__ps.x, __ps.y, __ps.z, 0) += value;

			// atomic_array_ref atomic(__view.at(__ps.x, __ps.y, __ps.z, __ps.layer));

			//sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::ext_intel_global_device_space>
			//	atomic(__view.at(__ps.x, __ps.y, __ps.z, __ps.layer % __ps.input.num_output_layers));

			sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::ext_intel_global_device_space>
				atomic(__view.at(__ps.x, __ps.y, __ps.z, 0));

			//__ps.input.num_output_layers

			//sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>
			//	atomic(__view.at(__ps.x, __ps.y, __ps.z, 0));


			//atomic_array_ref atomic(__view.at(__ps.x, __ps.y, __ps.z, 0));

			atomic.fetch_add(value);
		}

	public:

		weight_tracker(double weight, PhotonStruct& ps, matrix_view_adaptor<T>& view) :
			__weight{ weight }, __ps{ ps }, __view{ view }
		{;}

		weight_tracker<T>& operator=(const double& value)
		{
			__weight = value;

			return *this;
		}

		weight_tracker<T>& operator+=(const double& value)
		{
			__track(value);

			return *this;
		}

		weight_tracker<T>& operator-=(const double& value)
		{
			__track(value);

			__weight -= value;

			return *this;
		}

		weight_tracker<T>& operator/=(const double& value)
		{
			__weight /= value;

			return *this;
		}

		weight_tracker<T>& operator*=(const double& value)
		{
			__weight *= value;

			return *this;
		}

		inline double operator*(const double& value) const
		{
			return __weight * value;
		}

		inline bool operator==(const double& value) const
		{
			return __weight == value;
		}

		inline bool operator!=(const double& value) const
		{
			return __weight != value;
		}

		inline bool operator<(const double& value) const
		{
			return __weight < value;
		}
	};

	double x{ 0 }, y{ 0 }, z{ 0 };    // vector of position
	double ux{ 0 }, uy{ 0 }, uz{ 0 }; // vector of direction

	weight_tracker<float> w;

	bool dead{ false };

	size_t layer{ 0 };

	double sleft{ 0 };
	double step_size{ 0 };

	mcg59_t &random;

	const InputStruct& input;

	const LayerStruct* layerspecs;

	PhotonStruct(mcg59_t& random, matrix_view_adaptor<float> view, const InputStruct& input) :
		w{ 0.0, *this, view }, random{ random }, input{ input }, layerspecs{ input.layerspecs }
	{;}

	~PhotonStruct() = default;

	void init(const double Rspecular)
	{
		w = 1.0 - Rspecular;
		dead = 0;
		layer = 1; // LAYER CHANGE
		step_size = 0;
		sleft = 0;

		x = 0.0; // COORD CHANGE
		y = 0.0;
		z = 0.1;

		ux = 0.0;
		uy = 0.0;
		uz = 1.0;

		if ((layerspecs[1].mua == 0.0) && (layerspecs[1].mus == 0.0))
		{
			layer = 2; // LAYER CHANGE
			z = layerspecs[2].z0;
		}
	}

	void spin(const double anisotropy)
	{
		const double ux = this->ux;
		const double uy = this->uy;
		const double uz = this->uz;

		const double cost = SpinTheta(anisotropy);
		const double sint = sycl::sqrt(1.0 - cost * cost);

		const double psi = 2.0 * PI * get_random();

		const double cosp = sycl::cos(psi);


		double sinp; // = std::sin(psi);

		if (psi < PI)
		{
			sinp = sycl::sqrt(1.0 - cosp * cosp);
		}
		else
		{
			sinp = -sycl::sqrt(1.0 - cosp * cosp);
		}

		if (std::abs(uz) > COSZERO)
		{
			const double temp = (uz >= 0.0) ? 1.0 : -1.0;

			this->ux = sint * cosp;
			this->uy = sint * sinp;
			this->uz = temp * cost;
		}
		else
		{
			const double temp = sycl::sqrt(1.0 - uz * uz);

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

		w += MIN_WEIGHT;
	}

	void step_size_in_glass()
	{
		const auto& olayer = get_current_layer();

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
		const auto& olayer = get_current_layer();

		double dl_b;

		if (uz > 0.0)
		{
			dl_b = (olayer.z1 - z) / uz;
		}
		else if (uz < 0.0)
		{
			dl_b = (olayer.z0 - z) / uz;
		}

		if (uz != 0.0 && step_size > dl_b)
		{
			const double mut = olayer.mua + olayer.mus;

			sleft = (step_size - dl_b) * mut;
			step_size = dl_b;

			return true;
		}
		else
		{
			return false;
		}
	}

	void roulette()
	{
		if (w != 0.0 && get_random() < CHANCE)
		{
			w /= CHANCE;
		}
		else
		{
			dead = true;
		}
	}

	void record_r(double reflectance)
	{
		size_t ir, ia;

		ir = static_cast<size_t>(sycl::sqrt(x * x + y * y) / input.dr);
		ir = std::min<size_t>(ir, input.nr - 1);

		ia = static_cast<size_t>(std::acos(-uz) / input.da);
		ia = std::min<size_t>(ia, input.na - 1);

		w *= reflectance;
	}

	void record_t(double reflectance)
	{
		size_t ir, ia;

		ir = static_cast<size_t>(sycl::sqrt(x * x + y * y) / input.dr);
		ir = std::min<size_t>(ir, input.nr - 1);

		ia = static_cast<size_t>(std::acos(uz) / input.da);
		ia = std::min<size_t>(ia, input.na - 1);

		w *= reflectance;
	}

	void drop()
	{
		double dwa;
		size_t iz, ir;
		double mua, mus;

		iz = static_cast<size_t>(z / input.dz);
		iz = std::min<size_t>(iz, input.nz - 1);

		ir = static_cast<size_t>(sycl::sqrt(x * x + y * y) / input.dr);
		ir = std::min<size_t>(ir, input.nr - 1);

		const auto& olayer = get_current_layer();

		mua = olayer.mua;
		mus = olayer.mus;

		dwa = w * mua / (mua + mus);

		w -= dwa;
	}

	void cross_up_or_not()
	{
		double uz1 = 0.0;
		double r = 0.0;
		double ni = layerspecs[layer].n;
		double nt = layerspecs[layer - 1].n;

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
				record_r(0.0);
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
		double uz1 = 0.0;
		double r = 0.0;
		double ni = layerspecs[layer].n;
		double nt = layerspecs[layer + 1].n;

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

	double SpinTheta(const double anisotropy)
	{
		double cost;

		if (anisotropy == 0.0)
		{
			cost = 2.0 * get_random() - 1.0;
		}
		else
		{
			const double temp = (1.0 - anisotropy * anisotropy) / (1.0 - anisotropy + 2.0 * anisotropy * get_random());

			cost = (1 + anisotropy * anisotropy - temp * temp) / (2.0 * anisotropy);

			if (cost < -1.0)
			{
				cost = -1.0;
			}
			else if (cost > 1.0)
			{
				cost = 1;
			}
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
			const double mua = olayer.mua;
			const double mus = olayer.mus;

			if (sleft == 0.0)
			{
				double rnd;

				do
				{
					rnd = get_random();
				} while (rnd <= 0.0);

				step_size = -sycl::log(rnd) / (mua + mus);
			}
			else
			{
				step_size = sleft / (mua + mus);
				sleft = 0.0;
			}

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

	inline double get_random()
	{
		return random.next();
	}
};

double Rspecular(LayerStruct* Layerspecs_Ptr)
{
	double r1;
	double r2;
	double temp;

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
