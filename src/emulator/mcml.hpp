#pragma once
#include <cmath>
#include <random>
#include <CL/sycl.hpp>
#include <oneapi/dpl/random>
#include "../common/matrix.hpp"
#include "../common/iofile.hpp"

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
	LayerStruct* layerspecs{ nullptr };	/* layer parameters. */


	InputStruct() = default;

	InputStruct(const InputStruct&) = default;

	void free()
	{
		if (layerspecs)
		{
			::free(layerspecs);
		}
	}
};

double RFresnel(double n1, double n2, double ca1, double& ca2_Ptr)
{
	double r;

	if (n1 == n2)
	{
		ca2_Ptr = ca1;
		r = 0.0;
	}
	else if (ca1 > COSZERO)
	{
		ca2_Ptr = ca1;
		r = (n2 - n1) / (n2 + n1);
		r *= r;
	}
	else if (ca1 < COS90D)
	{
		ca2_Ptr = 0.0;
		r = 1.0;
	}
	else
	{
		double sa1, sa2;
		double ca2;

		sa1 = std::sqrt(1.0 - ca1 * ca1);
		sa2 = n1 * sa1 / n2;

		if (sa2 >= 1.0)
		{
			ca2_Ptr = 0.0;
			r = 1.0;
		}
		else
		{
			double cap, cam;
			double sap, sam;

			ca2_Ptr = ca2 = std::sqrt(1.0 - sa2 * sa2);

			cap = ca1 * ca2 - sa1 * sa2; /* c+ = cc - ss. */
			cam = ca1 * ca2 + sa1 * sa2; /* c- = cc + ss. */
			sap = sa1 * ca2 + ca1 * sa2; /* s+ = sc + cs. */
			sam = sa1 * ca2 - ca1 * sa2; /* s- = sc - cs. */

			r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
		}
	}

	return r;
}

struct PhotonStruct
{
	double x{ 0 }, y{ 0 }, z{ 0 };
	double ux{ 0 }, uy{ 0 }, uz{ 0 };
	double w{ 0 };

	bool dead{ false };

	size_t layer{ 0 };

	double sleft{ 0 };
	double step_size{ 0 };

	 oneapi::dpl::minstd_rand engine;
	 oneapi::dpl::uniform_real_distribution<double> distr;

	const InputStruct& input;

	const LayerStruct* layerspecs;

	matrix_view_adaptor<float> view;

	PhotonStruct(matrix_view_adaptor<float> view, const InputStruct& input, const LayerStruct* l) :
		engine{0, 100}, distr{ 0.0, 1.0 }, input{ input }, layerspecs{ l }, view(view)
	{;}

	~PhotonStruct() = default;

	void track(float weight = 1.0F)
	{
		auto v = atomic_array_ref(view.at(this->x, this->y, this->z, this->layer));

		v.fetch_add(weight);
	}


	void init(const double Rspecular)
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

		track();
	}

	void spin(const double anisotropy)
	{
		const double ux = this->ux;
		const double uy = this->uy;
		const double uz = this->uz;

		const double cost = SpinTheta(anisotropy);
		const double sint = std::sqrt(1.0 - cost * cost);

		const double psi = 2.0 * PI * get_random();

		const double cosp = std::cos(psi);
		const double sinp = setsign<double, uint64_t>(std::sqrt(1.0 - cosp * cosp), psi < PI);

		if (fabs(uz) > COSZERO)
		{
			this->ux = sint * cosp;
			this->uy = sint * sinp;
			this->uz = cost * sgn(uz);
		}
		else
		{
			const double temp = std::sqrt(1.0 - uz * uz);

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

		track();
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

		const bool hit = (uz != 0.0 && step_size > dl_b);

		if (hit)
		{
			sleft = (step_size - dl_b) * (olayer.mua + olayer.mus);
			step_size = dl_b;
		}

		return hit;
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

		ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
		ir = std::min<size_t>(ir, input.nr - 1);

		ia = static_cast<size_t>(std::acos(-uz) / input.da);
		ia = std::min<size_t>(ia, input.na - 1);

		w *= reflectance;
	}

	void record_t(double reflectance)
	{
		size_t ir, ia;

		ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
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

		ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
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

				track();
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

				track();
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

				step_size = -std::log(rnd) / (mua + mus);
			}
			else
			{
				step_size = sleft / (mua + mus);
				sleft = 0.0;
			}

			const auto hit = hit_boundary();

			hop();

			if (hit)
			{
				cross_or_not();
			}
			else
			{
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
		return distr(engine);
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

void configure_input(InputStruct& input)
{
	input.num_photons = 102400;
	input.Wth = 0.0001;
	input.dz = 0.01;
	input.dr = 0.02;
	input.da = 0.015707;
	input.nz = 200;
	input.nr = 500;
	input.na = 100;
	input.num_layers = 5;
}

void configure(LayerStruct *layerspecs, sycl::queue q)
{
	layerspecs[0].z0 = std::numeric_limits<double>::min();
	layerspecs[0].z1 = std::numeric_limits<double>::min();
	layerspecs[0].n = 1.0;
	layerspecs[0].mua = std::numeric_limits<double>::min();
	layerspecs[0].mus = std::numeric_limits<double>::min();
	layerspecs[0].anisotropy = std::numeric_limits<double>::min();
	layerspecs[0].cos_crit0 = std::numeric_limits<double>::min();
	layerspecs[0].cos_crit1 = std::numeric_limits<double>::min();

	layerspecs[1].z0 = 0;
	layerspecs[1].z1 = 0.01;
	layerspecs[1].n = 1.5;
	layerspecs[1].mua = 4.3;
	layerspecs[1].mus = 107;
	layerspecs[1].anisotropy = 0.79;
	layerspecs[1].cos_crit0 = 0.745355;
	layerspecs[1].cos_crit1 = 0.35901;

	layerspecs[2].z0 = 0.01;
	layerspecs[2].z1 = 0.03;
	layerspecs[2].n = 1.4;
	layerspecs[2].mua = 2.7;
	layerspecs[2].mus = 187.0;
	layerspecs[2].anisotropy = 0.82;
	layerspecs[2].cos_crit0 = 0.0;
	layerspecs[2].cos_crit1 = 0.0;

	layerspecs[3].z0 = 0.03;
	layerspecs[3].z1 = 0.05;
	layerspecs[3].n = 1.4;
	layerspecs[3].mua = 3.3;
	layerspecs[3].mus = 192.0;
	layerspecs[3].anisotropy = 0.82;
	layerspecs[3].cos_crit0 = 0.0;
	layerspecs[3].cos_crit1 = 0.0;

	layerspecs[4].z0 = 0.05;
	layerspecs[4].z1 = 0.14;
	layerspecs[4].n = 1.4;
	layerspecs[4].mua = 2.7;
	layerspecs[4].mus = 187.0;
	layerspecs[4].anisotropy = 0.82;
	layerspecs[4].cos_crit0 = 0.0;
	layerspecs[4].cos_crit1 = 0.0;

	layerspecs[5].z0 = 0.14;
	layerspecs[5].z1 = 0.2;
	layerspecs[5].n = 1.4;
	layerspecs[5].mua = 2.4;
	layerspecs[5].mus = 194;
	layerspecs[5].anisotropy = 0.82;
	layerspecs[5].cos_crit0 = 0.0;
	layerspecs[5].cos_crit1 = 0.6998542;

	layerspecs[6].z0 = std::numeric_limits<double>::min();
	layerspecs[6].z1 = std::numeric_limits<double>::min();
	layerspecs[6].n = 1.0;
	layerspecs[6].mua = std::numeric_limits<double>::min();
	layerspecs[6].mus = std::numeric_limits<double>::min();
	layerspecs[6].anisotropy = std::numeric_limits<double>::min();
	layerspecs[6].cos_crit0 = std::numeric_limits<double>::min();
	layerspecs[6].cos_crit1 = std::numeric_limits<double>::min();
}
