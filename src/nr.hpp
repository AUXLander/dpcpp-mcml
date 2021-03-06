#pragma once

#include "data/tracker.h"
#include "io/io.hpp"

#include "math/math.h"

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

	LayerStruct(const LayerStruct& o):
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

double Rspecular(LayerStruct* Layerspecs_Ptr);


//struct PhotonStruct
//{
//	double x{0}, y{0}, z{0};
//	double ux{0}, uy{0}, uz{0};
//	double w{0};
//
//	bool dead {false};
//
//	size_t layer{0};
//
//	double step_size{0};
//	double sleft{0};
//
//	const InputStruct& input;
//
//	OutStruct& output;
//
//	tracker::local_thread_storage track;
//
//	PhotonStruct(const InputStruct&, OutStruct&);
//
//	~PhotonStruct()
//	{
//		auto& g = tracker::instance();
//
//		g.track(std::move(track));
//	}
//
//	void init(const double Rspecular, LayerStruct* Layerspecs_Ptr);
//
//	void spin(const double anisotropy);
//
//	void hop();
//
//	void step_size_in_glass();
//	bool hit_boundary();
//	void roulette();
//	void record_r(double Refl);
//	void record_t(double Refl);
//	void drop();
//
//	void cross_up_or_not();
//	void cross_down_or_not();
//
//	void cross_or_not();
//
//	void hop_in_glass();
//	void hop_drop_spin();
//
//	LayerStruct& get_current_layer()
//	{
//		assert(input.layerspecs);
//
//		return input.layerspecs[layer];
//	}
//};
//
//struct PhotonStruct
//{
//	double x{0}, y{0}, z{0};
//	double ux{0}, uy{0}, uz{0};
//	double w{0};
//
//	bool dead {false};
//
//	size_t layer{0};
//
//	double step_size{0};
//	double sleft{0};
//
//	const InputStruct& input;
//
//	const access_output<double>& output;
//
//	PhotonStruct(const InputStruct&, const access_output<double>&) :
//		input{ input }, output{ output }
//	{;}
//
//	~PhotonStruct() {;}
//
//	void init(const double Rspecular, LayerStruct* Layerspecs_Ptr);
//
//	void spin(const double anisotropy);
//
//	void hop();
//
//	void step_size_in_glass();
//	bool hit_boundary();
//	void roulette();
//	void record_r(double Refl);
//	void record_t(double Refl);
//	void drop();
//
//	void cross_up_or_not();
//	void cross_down_or_not();
//
//	void cross_or_not();
//
//	SYCL_EXTERNAL void hop_in_glass();
//	void hop_drop_spin();
//
//	LayerStruct& get_current_layer()
//	{
//		assert(input.layerspecs);
//
//		return input.layerspecs[layer];
//	}
//};