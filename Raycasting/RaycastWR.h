#pragma once
#include "Header.h"

#include <vector>
#include <string>
#include <fstream>

#include "RaycastSystem.h"
#include "ScalarGrid.h"

#include <omp.h>

// ========================================================================
// Payload data
template<typename T>
struct PayloadWR 
{
	struct AreaBasedAmplitude2D 
	{
		// initial amplitude to the left (previous in the structure) of the ray
		T areaLeft;
		// initial amplitude to the right (next in the structure) of the ray
		T areaRight;
		// initial amplitude
		T amplitude;
		// computed amplitude
		T computed_amplitude;
	};

	std::vector<AreaBasedAmplitude2D> ampData;
};

// ========================================================================
// Declarations

#pragma region header defines

// Large template of template<> header with defaults explicitly set. Use it with child class' DECLARATION right before the name of the custom class
#define RAYCAST_SYSTEM_TEMPLATE_HEADER_DEFAULTS \
template\
<\
	typename T = Float, \
	template<typename> class RayTemplate = Ray2D, \
	template<typename, typename> class IntegratorTemplate = IntegratorEuler, \
	typename PayloadT = PayloadWR<T>\
>

// Large template of template<> header without defaults. Use it with child class' DEFINITION before the function definition
#define RAYCAST_SYSTEM_TEMPLATE_HEADER \
template\
<\
	typename T, \
	template<typename> class RayTemplate, \
	template<typename, typename> class IntegratorTemplate, \
	typename PayloadT\
>

#pragma endregion

RAYCAST_SYSTEM_TEMPLATE_HEADER_DEFAULTS
class RaycastWR : public RaycastSystem<RAYCAST_SYSTEM_TEMPLATE_PARAMS> 
{
public:
	RAYCAST_SYSTEM_INHERIT_FIELDS;

	virtual void Init() override;
	virtual void PreStep() override;
	virtual void PostStep() override;
	virtual void EvaluateDeltaTime() override;
	virtual T C(const SubT& x) const override;
	virtual T C_log_dx(const SubT& x, int j) const override;

	bool LoadScalarGrid(std::string& filename, uint sizeX, uint sizeY, SubT stepVec);
	// Arg @asIs passes grid as is without modifications
	bool LoadScalarGrid(const ScalarGrid<T> other, SubT in_stepVec, bool asIs = false);

	// Evaluate amplitudes, stored in payload, on demand
	void ComputeAmplitudes();

	bool bInsertRays = false;
	T max_distance_parting = T(0.005);
	T collapse_distance = T(0.0025);
	T amplitude_penalty = T(0.05);
	bool bEnclosedShape = true;
	bool bEvaluateDeltaTime = false;
	bool bFastDifferential = true;
	int smoothingSteps = 30;

	ScalarGridSamplerType samplerType = ScalarGridSamplerType::Cubic;

private:
	ScalarGrid<T> scalarGrid;
	ScalarGrid<T> logGrid;
	ScalarGrid<T> diffGridX;
	ScalarGrid<T> diffGridY;
	ScalarGrid<T> diffGridZ;
	bool bScalarGridLoaded = false;
	SubT stepVec;
	T maxStepDim;
};




// ==========================================================================
// Definitions

RAYCAST_SYSTEM_TEMPLATE_HEADER
void RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::Init()
{
	// Fill payload
	current.payload.ampData.resize(current.rayData.size());

	auto dataSize = current.rayData.size();

	for (int i = 0; i < dataSize; ++i)
	{
		int i_prev = (i + dataSize - 1) % dataSize;
		int i_this = i;
		int i_next = (i + 1) % dataSize;

		current.payload.ampData[i] =
		{	.areaLeft = (current.rayData[i_prev].x() - current.rayData[i_this].x()).norm(),
			.areaRight = (current.rayData[i_this].x() - current.rayData[i_next].x()).norm(),
			.amplitude = T(1) 
		};
	}
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
void RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::PreStep() 
{
	if (!bInsertRays)
		return;

	// Since rays are being inserted their count can go to outer space, 
	// hence after the insertion some rays should be collapsed if they go too dense

	T max_sq = max_distance_parting * max_distance_parting;
	T min_sq = collapse_distance * collapse_distance;

	std::vector<RayT> insVect;
	std::vector<typename PayloadT::AreaBasedAmplitude2D> insVect_payload;
	insVect.reserve(5); // Preallocate some, might be slower than allocating on demand though
	insVect_payload.reserve(5);

	//auto dataSize = current.rayData.size();
	//int loop_end = bEnclosedShape ? int(dataSize) : int(dataSize - 1);
	for (int loop_i = 0; loop_i < (bEnclosedShape ? (current.rayData.size()) : (current.rayData.size() - 1)); ++loop_i)
	{
		int i0 = loop_i;
		int i1 = (i0 + 1) % current.rayData.size();

		SubT x0 = current.rayData[i0].x();
		SubT x1 = current.rayData[i1].x();
		T dist_sq = (x1 - x0).squaredNorm();

		if (current.payload.ampData[i0].computed_amplitude >= amplitude_penalty && current.payload.ampData[i0].computed_amplitude >= amplitude_penalty)
		if (max_sq < dist_sq && min_sq < max_sq) // If for some reason min distance is greater than max, do not try to insert, collapse instead
		{
			T dist = std::sqrt(dist_sq);
			uint ins_num = uint(std::ceil(dist / max_distance_parting));

			insVect.resize(ins_num); // Set size on top of preallocated memory
			insVect_payload.resize(ins_num);

			for (int i = 0; i < int(ins_num); ++i)
			{
				T alpha = T(1.0) - (T(i + 1) / (ins_num + 1));
				RayT ray = VecT(current.rayData[i0]) * alpha + VecT(current.rayData[i1]) * (1.0 - alpha);
				insVect[i] = ray;
				insVect_payload[i].areaLeft = insVect_payload[i].areaRight = current.payload.ampData[i0].areaRight / (ins_num + 1);
				insVect_payload[i].amplitude = T(1);
			}

			current.rayData.insert(current.rayData.begin() + i1, insVect.begin(), insVect.end());
			current.payload.ampData[i0].areaRight /= (ins_num + 1);
			current.payload.ampData[i1].areaLeft /= (ins_num + 1);
			current.payload.ampData.insert(current.payload.ampData.begin() + i1, insVect_payload.begin(), insVect_payload.end());

			loop_i += ins_num - 1;
		}

		// Collapse without else statement, which basically means collapsing overrides insertion
		if (dist_sq < min_sq && current.rayData.size() >= 3)
		{
			if (i0 > i1)
				__debugbreak();
			// Use average, give 0 fucks about slowness vector change, which can come bite in the ass later
			RayT ray = (VecT(current.rayData[i0]) + VecT(current.rayData[i1])) * T(0.5);
			// Erase the second one in the pair and replace the first one
			current.rayData.erase(current.rayData.begin() + i1);
			current.rayData[i0] = ray;

			// Handle payload as well
			auto& ampData = current.payload.ampData;

			T addedArea = ampData[i0].areaRight * T(0.5);
			ampData.erase(current.payload.ampData.begin() + i1);

			auto dataSize_payload = current.payload.ampData.size();
			ampData[i0].areaLeft += addedArea;
			ampData[i0].areaRight += addedArea;
			ampData[(i0 + dataSize_payload - 2) % dataSize_payload].areaRight += addedArea;
			ampData[(i0 + 1) % dataSize_payload].areaLeft += addedArea;

			// Go back 2 if possible, because technically the context for neightbouring rays also changes.
			if (i0 > 0)
				loop_i -= 1;
			else
				--loop_i;
		}
	}
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
void RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::PostStep() 
{
	ComputeAmplitudes();

	// Normalize velocities
#pragma omp parallel for
	for (int i = 0; i < current.rayData.size(); ++i) 
	{
		T vel = C(current.rayData[i].x());
		SubT slowness = current.rayData[i].p();
		slowness = slowness.normalized() / vel;
		current.rayData[i].p(slowness);
	}
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
T RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::C(const RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::SubT& x) const
{
	if (bScalarGridLoaded) 
	{
		//return T(100);
		return scalarGrid.Sample(samplerType, x.x() / stepVec.x(), x.y() / stepVec.y());
	}
	else
		return __super::C(x) * 10;
	for (int i = 0; i < SubT::RowsAtCompileTime; ++i)
		if (std::abs(0.5 - x(i)) > T(0.25))
			return T(1.0);
	return T(0.5);
	/*T coordSum = 0;
	for (int i = 0; i < SubT::RowsAtCompileTime; ++i)
		coordSum += std::max(std::abs(x(i) - T(0.5)) - T(0.25), T(0));
	coordSum = std::min(coordSum * 50000, 0.5) + 0.5;
	return coordSum;*/
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
T RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::C_log_dx(const RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::SubT& x, int j) const
{
	if (bScalarGridLoaded)
	{
		if (!bFastDifferential)
			return logGrid.SampleDerivative(samplerType, j, x.x() / stepVec.x(), x.y() / stepVec.y()) / stepVec(j);
		
		switch (j)
		{
		default:
		case 0:
			return diffGridX.Sample(samplerType, x.x() / stepVec.x(), x.y() / stepVec.y()) / stepVec.x();
		case 1:
			return diffGridY.Sample(samplerType, x.x() / stepVec.x(), x.y() / stepVec.y()) / stepVec.y();
		//case 2:
		//	return diffGridZ.Sample(samplerType, x.x() / stepVec.x(), x.y() / stepVec.y()) / stepVec.z();
		}
	}
	else
		return __super::C_log_dx(x, j);
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
inline bool RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::LoadScalarGrid(std::string& filename, uint sizeX, uint sizeY, SubT in_stepVec)
{
	ifstream inf(filename);
	if (!inf.is_open()) 
	{
		std::cerr << "Failed to open the file " << filename << std::endl;
		bScalarGridLoaded = false;
		return false;
	}
	inf.close();

	bScalarGridLoaded = scalarGrid.ReadFromBinary(filename, sizeX, sizeY);
	stepVec = in_stepVec;
	maxStepDim = stepVec.lpNorm<Eigen::Infinity>();
	for (int i = 0; i < smoothingSteps; ++i) scalarGrid.ExpKernel();

	logGrid = scalarGrid;
	logGrid.Process([](T v, int x, int y, int z)->T { return std::log(v); });

	if (bFastDifferential)
	{
		diffGridX = logGrid;
		diffGridX.CentralDifference(0);

		diffGridY = logGrid;
		diffGridY.CentralDifference(1);
	}

	return bScalarGridLoaded;
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
bool RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::LoadScalarGrid(const ScalarGrid<T> other, SubT in_stepVec, bool asIs)
{
	if (other.GetSize(0) == 0 || other.GetSize(1) == 0)
		return false;

	stepVec = in_stepVec;
	maxStepDim = stepVec.lpNorm<Eigen::Infinity>();
	scalarGrid = other;

	if (!asIs)
	for (int i = 0; i < smoothingSteps; ++i) scalarGrid.ExpKernel();

	logGrid = scalarGrid;
	logGrid.Process([](T v, int x, int y, int z)->T { return std::log(v); });

	if (bFastDifferential)
	{
		diffGridX = logGrid;
		diffGridX.CentralDifference(0);

		diffGridY = logGrid;
		diffGridY.CentralDifference(1);
	}

	bScalarGridLoaded = true;

	return true;
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
inline void RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::ComputeAmplitudes()
{
	if (current.rayData.size() < 3)
		return;
#pragma omp parallel for
	for (int loop_i = 0; loop_i < current.rayData.size(); ++loop_i)
	{
		T areaRatio = T(0);
		if (!(bEnclosedShape || loop_i == 0)) 
		{
			areaRatio +=
				current.payload.ampData[loop_i].areaLeft / 
				(current.rayData[loop_i - 1].x() - current.rayData[loop_i].x()).norm();
		}
		if (!(bEnclosedShape || loop_i == current.rayData.size() - 1)) 
		{
			areaRatio +=
				current.payload.ampData[loop_i].areaRight / 
				(current.rayData[loop_i].x() - current.rayData[loop_i + 1].x()).norm();
		}
		current.payload.ampData[loop_i].computed_amplitude = areaRatio * current.payload.ampData[loop_i].amplitude;
	}
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
void RaycastWR<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::EvaluateDeltaTime() 
{
	if (!bEvaluateDeltaTime || !bScalarGridLoaded)
		return;

	T maxScalar = 0.;

#pragma omp parallel for
	for (int i = 0; i < current.rayData.size(); ++i)
	{
		T localMaxScalar = std::max(C(current.rayData[i].x()), maxScalar);
#pragma omp critical
		maxScalar = std::max(localMaxScalar, maxScalar);
	}

	// Correct timestep to fit scalar grid value.
	deltaTime = maxStepDim / maxScalar * T(0.5);
}