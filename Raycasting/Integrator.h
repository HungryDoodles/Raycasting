#pragma once
#include "Header.h"

#include <functional>
#include <atomic>
#include <mutex>

#include <Geometry>

/* 
Class that blindly performs integration
T is scalar type
VecT is algebraic vector type (preferably having the same scalar type)
*/
template<typename T, typename VecT>
class Integrator
{
public:
	// Initialize. Create containers if needed
	void Init() {};

	// Step before bulk integration happens. Initialize containers if needed
	void PreIntegration() {}

	// Bulk integration
	VecT Integrate(const VecT&, T Dt, std::function<VecT(const VecT&)> RHS) {}

	// Step after the bulk integration happens. 
	void PostIntegration() {}

	// Shows how bad the last integration is if the method allows it
	// If the value is above 1 the integration might have failed
	// and requires smaller steps
	inline thread_local static T last_relative_error = 0;
};

template<typename T, typename VecT>
class IntegratorEuler : public Integrator<T, VecT>
{
public:
	VecT Integrate(const VecT& old, T Dt, std::function<VecT(const VecT&)> RHS)
	{
		return old + RHS(old) * Dt;
	}
};

template<typename T, typename VecT>
class IntegratorRK4Expl : public Integrator<T, VecT>
{
public:
	VecT Integrate(const VecT& old, T Dt, std::function<VecT(const VecT&)> RHS)
	{
		VecT k1 = RHS(old);
		VecT k2 = RHS(old + Dt * 0.5 * k1);
		VecT k3 = RHS(old + Dt * 0.5 * k2);
		VecT k4 = RHS(old + Dt * k3);

		return old + 1 / 6. * Dt * (k1 + 2 * k2 + 2 * k3 + k4);
	}
};

template<typename T, typename VecT>
class IntegratorGaussLegendre : public Integrator<T, VecT>
{
public:
	inline static std::atomic<ulong> iterations_limit_stops = 0;
	inline static std::atomic<ulong> precision_limit_stops = 0;
	inline static std::atomic<ulong> total_iterations = 0;
	inline static std::atomic<ulong> total_calls = 0;
	inline static T max_error = 0;
	inline static std::mutex max_error_lock;

	inline static uint max_iterations_num = 50;
	inline static T target_precision = T(4e-8);
	inline static const T sqrt_3 = sqrt(3);

	VecT Integrate(const VecT& old, T Dt, std::function<VecT(const VecT&)> RHS)
	{
		// 1/4 | 1/4-1/6 sqrt(3)
		// 1/4+1/6 sqrt(3) | 1/4
		// 1/2 | 1/2

		// Using fixed point method...
		// x_n+1 = f(x_n)
		// ...against Gauss-Legendre system...
		// f(y_n + h * sum_j(aij*kj)) = ki

		// ...forming the coefficients for the step
		// y_n+1 = y_n + h * sum(bi * ki)

		// Initial guess
		VecT k1 = VecT::Zero();
		VecT k2 = VecT::Zero();

		// Begin iterations
		bool bPrecisionLimitStop = false;
		int i;
		T error;
		for (i = 0; i < int(max_iterations_num); ++i)
		{
			VecT new_k1 = RHS(old + Dt * (0.25 * k1 + (0.25 - sqrt_3/6) * k2));
			VecT new_k2 = RHS(old + Dt * ((0.25 + sqrt_3/6) * k1 + 0.25 * k2));
			error = max((k1 - new_k1).lpNorm<Eigen::Infinity>(), (k2 - new_k2).lpNorm<Eigen::Infinity>());
			k1 = new_k1;
			k2 = new_k2;
			if (error < target_precision)
			{
				bPrecisionLimitStop = true;
				//++precision_limit_stops;
				break;
			}
		}
		Integrator<T, VecT>::last_relative_error = error / target_precision;

		if (i == int(max_iterations_num))
			++iterations_limit_stops;
		if (bPrecisionLimitStop)
			++precision_limit_stops;
		total_iterations += i;
		++total_calls;
		
		{
			std::lock_guard<mutex> lock(max_error_lock);
			max_error = max(max_error, error);
		}

		return old + Dt * (0.5 * k1 + 0.5 * k2);
	}
};