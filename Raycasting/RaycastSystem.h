#pragma once
#include "Header.h"

#include <numeric>
#include <functional>
#include <string>
#include <fstream>
#include <thread>
#include <omp.h>

#include "Integrator.h"
#include "Ray.h"

// Crutch
struct EmptyPayload {};

// IMPORTANT!!!!!
// Defines are used to simplify the process of making a child class

// Large template of template<> header with defaults explicitly set. Use it with child class' DECLARATION right before the name of the custom class
#define RAYCAST_SYSTEM_TEMPLATE_HEADER_DEFAULTS \
template\
<\
	typename T = Float, \
	template<typename> class RayTemplate = Ray2D, \
	template<typename, typename> class IntegratorTemplate = IntegratorEuler, \
	typename PayloadT = EmptyPayload\
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

/* An unenclosed list of params for use to declare with raycast system class
Example:
template<RAYCAST_SYSTEM_TEMPLATE_HEADER>
void MyCustomRSType<RAYCAST_SYSTEM_TEMPLATE_PARAMS>::CustomFunction(args){...}
*/
#define RAYCAST_SYSTEM_TEMPLATE_PARAMS T, RayTemplate, IntegratorTemplate, PayloadT

// Put this inside the custom class after <public:> visibility specifier to inherit default typenames and fields
#define RAYCAST_SYSTEM_INHERIT_FIELDS \
using RST = RaycastSystem<T, RayTemplate, IntegratorTemplate, PayloadT>; \
using typename RST::SubT; \
using typename RST::VecT; \
using typename RST::RayT; \
using typename RST::Integrator; \
using RST::integrator; \
using RST::current; \
using RST::next; \
using RST::snapshots; \
using RST::deltaTime;



// Raycast System
//----------------------------------------------------------------------------------------------------------

RAYCAST_SYSTEM_TEMPLATE_HEADER_DEFAULTS
class RaycastSystem
{
public:

	// Types
	//========================================

	typedef RayTemplate<T> RayT;
	typedef typename RayT::VecT VecT;
	typedef typename RayT::SubT SubT;
	typedef IntegratorTemplate<T, VecT> Integrator;

	struct Snapshot
	{
		// When it happened
		T timePoint;
		// What it had
		std::vector<RayT> rayData;
		// Any custom information related to this snapshot, e.g. structure
		PayloadT payload;
	};

	RaycastSystem() {}
	virtual ~RaycastSystem() { worker = thread(); }

	// Control
	// =========================================

	// Set delta time. Ignored if deltaTime is being computed internally.
	virtual void SetDeltaTime(T newDeltaTime);

	// Creates N elements of RayT using generator var@rayGen
	virtual void Create(uint N, std::function<RayT(uint, RaycastSystem*)> rayGen);

	// Performs the simulation until the simulation time reaches var@Time or number of steps reaches var@max_steps
	virtual void Simulate(T Time, long long max_steps = std::numeric_limits<long long>::max());

	// Performs the simulation in a separate thread
	virtual void SimulateAsync(T Time, long long max_steps = std::numeric_limits<long long>::max());

	// Clears everything
	virtual void Reset();


	// System parameters
	// =========================================

	// Right-hand-side
	// Controls state change from previous var@state to new ret@val
	virtual VecT RHS(const VecT& state) const;

	// Wavespeed given at any point of the field. Exponential bell shape is used by default.
	// var@x indicates position of a ray's current sub-vector type
	virtual T C(const SubT& x) const;

	// Numerical derivative of 4-th order
	// var@j is a derivative dimension number: 0 = X, 1 = Y, 2 = Z (if exists)
	virtual T C_log_dx(const SubT& x, int j) const;

	// Data access
	// =========================================

	size_t GetSnapshotsNum() const;
	Snapshot* const GetCurrentSnapshot() const;
	Snapshot GetCurrentSnapshot();
	Snapshot* const GetSnapshotAt(size_t position) const;
	Snapshot* GetSnapshotAt(size_t position);

	// Creates in interruptor running off of the main worker thread.
	// Automatically cleans up the trail of solutions until only one left
	void Subscribe_OnSolution(std::function<void(int, Snapshot*)> on_solution);
	void Unsubscribe_OnSolution();

	// Misc
	// ==========================================
	virtual void Verbose(const std::string& filename, int skipping = 1);

	// Async data
	struct AsyncStatus 
	{
		T current_time;
		T time_step;
		long long steps_num;
		long long rays_num;
		bool bWorking;
	} asyncStatus;

	// The maximum number of time divisions allowed if the chosen integrator unable approach desired precision.
	// The number of actual substeps performed corresponds to the power of 2 of this variable's value
	int maxSubstepping = 0;

protected:

	// The function that performs steps inside of the sync simulation
	// Needed to avoid the compiler bug with the templates
	virtual int AsyncWorker(T Time, long long max_steps);

	// The function reliable for simulation advancement in a form of a single step
	virtual void Step();

	// Event that happens at the system start
	virtual void Init() {}
	// Evaluate deltaTime 
	virtual void EvaluateDeltaTime() { }
	// Event that fires before the step happens
	virtual void PreStep() {}
	// Event that fires after the step happens
	virtual void PostStep() {}

	// Integrator class responsible for blind integration of a single RayT::VecT by the means of calling RHS in the required order
	Integrator integrator;

	// Current snapshot of the system at this very time
	Snapshot current;
	// Next snapshot that the step simulation data will be written to
	Snapshot next;
	// All the records of all snapshots
	std::vector<Snapshot> snapshots;

	// DeltaTime chosen for this step.
	// Designed to be set on EvaluateDeltaTime
	T deltaTime = T(0.1);

private:
	std::thread worker;

	std::function<void(int, Snapshot*)> solutionEvent;
	bool bTriggerOnSolutionEvent = false;
};



// Hack
#define RS_TEMPL RaycastSystem<RAYCAST_SYSTEM_TEMPLATE_PARAMS>

// ==================================================================
// Class member definitions

RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::SetDeltaTime(T newDeltaTime)
{
	if (newDeltaTime > 0)
		deltaTime = newDeltaTime;
}

// Creates N elements of RayT using generator var@rayGen
RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::Create(uint N, std::function<RayT(uint, RaycastSystem*)> rayGen)
{
	Reset();
	if (N < 1) return;
	current.rayData.resize(N);
	for (int i = 0; i < int(N); ++i)
		current.rayData[i] = rayGen(i, this);
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
int RS_TEMPL::AsyncWorker(T Time, long long max_steps) 
{
	T& time = current.timePoint;
	long long numSteps = 0;
	while (time <= Time || numSteps >= max_steps)
	{
		numSteps++;
		EvaluateDeltaTime();
		asyncStatus.time_step = deltaTime;
		PreStep();
		next.rayData.resize(current.rayData.size());
		next.payload = current.payload;

		Step();

		snapshots.push_back(next);
		PostStep();
		current = next;

		asyncStatus.current_time = time;
		asyncStatus.steps_num = numSteps;
		asyncStatus.rays_num = current.rayData.size();

		if (bTriggerOnSolutionEvent)
		{
			while (!snapshots.empty())
			{
				Snapshot* snapshot = &(snapshots.front());
				solutionEvent(numSteps, snapshot);
				snapshots.erase(snapshots.begin());
			}
		}
	}
	asyncStatus.bWorking = false;

	return 0;
}

// The function reliable for simulation advancement in a form of a single step.
// Must rely on var@current state and write into var@next
RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::Step()
{
#pragma omp parallel for schedule(dynamic, 20)
	for (int i = 0; i < current.rayData.size(); ++i)
	{
		int substepping = 0;

	restart_stepping:
		auto state = current.rayData[i];
		decltype(integrator)::last_relative_error = 0;
		for (int j = 0; j < (1 << substepping); ++j)
		{
			integrator.PreIntegration();
			state = integrator.Integrate
			(
				state,
				deltaTime / (1 << substepping),
				std::bind(&RaycastSystem::RHS, this, std::placeholders::_1)
			);
			integrator.PostIntegration();

			if (substepping < maxSubstepping)
			if (decltype(integrator)::last_relative_error > T(1.1))
			{
				substepping++;
				//cout << "Substepping: " << substepping << endl;
				goto restart_stepping;
			}
		}
		next.rayData[i] = state;
	}
	next.timePoint = current.timePoint + deltaTime;
}

// Performs the simulation until the simulation time reaches var@Time or number of steps reaches var@max_steps
RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::Simulate(T Time, long long max_steps)
{
	Init();
	integrator.Init();
	snapshots.push_back(current);
	T& time = current.timePoint;
	long long numSteps = 0;
	while (time <= Time || numSteps >= max_steps)
	{
		numSteps++;
		EvaluateDeltaTime();
		PreStep();
		next.rayData.resize(current.rayData.size());
		next.payload = current.payload;

		Step();

		snapshots.push_back(next);
		PostStep();
		current = next;
		cout << "T = " << time << endl;

		if (bTriggerOnSolutionEvent) 
		{
			while (!snapshots.empty())
			{
				Snapshot* snapshot = &(snapshots.front());
				solutionEvent(numSteps, snapshot);
				snapshots.erase(snapshots.begin());
			}
		}
	}
}

// Performs the simulation in a separate thread
RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::SimulateAsync(T Time, long long max_steps)
{
	Init();
	integrator.Init();
	snapshots.push_back(current);

	asyncStatus.bWorking = true;
	worker = std::thread(&RaycastSystem::AsyncWorker, this, Time, max_steps);
	worker.detach();
}

// Clears everything
RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::Reset()
{
	snapshots.clear();
	current.rayData.clear();
	next.rayData.clear();
	current.timePoint = 0;
}

// Right-hand-side
// Controls state change from previous var@state to new ret@val
RAYCAST_SYSTEM_TEMPLATE_HEADER
typename RS_TEMPL::VecT RS_TEMPL::RHS(const typename RS_TEMPL::VecT& state) const
{
	auto rowsNum = RayT::SubT::RowsAtCompileTime;
	Eigen::Map<SubT> x((T*)state.data(), rowsNum);
	Eigen::Map<SubT> p((T*)state.data() + rowsNum, rowsNum);

	T c_val = C(x);
	SubT log_dx_vec;
	for (int i = 0; i < rowsNum; ++i)
		log_dx_vec(i) = C_log_dx(x, i);
	

	SubT dx = c_val * c_val * p;
	SubT dp = -log_dx_vec;

	VecT value;
	memcpy(value.data() + 0, dx.data(), rowsNum * sizeof(T));
	memcpy(value.data() + rowsNum, dp.data(), rowsNum * sizeof(T));
	return value;
}

// Wavespeed given at any point of the field. Exponential bell shape is used by default.
// var@x indicates position of a ray's current sub-vector type
RAYCAST_SYSTEM_TEMPLATE_HEADER
T RS_TEMPL::C(const RS_TEMPL::SubT& x) const
{
	SubT x0 = SubT::Ones() * T(0.5);
	T A = T(1.0);
	T B = T(0.25);
	T C = -100;

	return A - B * exp((x0 - x).squaredNorm() * C);
}

// Numerical derivative of 4-th order
// var@j is a derivative dimension number: 0 = X, 1 = Y, 2 = Z (if exists)
RAYCAST_SYSTEM_TEMPLATE_HEADER
T RS_TEMPL::C_log_dx(const RS_TEMPL::SubT& x, int j) const
{
	if (j < 0 || j >= SubT::RowsAtCompileTime)
		return T(0);

	SubT dx = SubT::Zero();
	dx[j] = d_h;
	auto C_inv_sq = [this](SubT x)->T { return -std::log(C(x)); };
	T num =
		- C_inv_sq(x - dx * 2.)
		+ 8 * C_inv_sq(x - dx)
		- 8 * C_inv_sq(x + dx)
		+ C_inv_sq(x + dx * 2.);
	return num / (12 * d_h);
}

// Data access
// =========================================

RAYCAST_SYSTEM_TEMPLATE_HEADER
size_t RS_TEMPL::GetSnapshotsNum() const
{
	return snapshots.size();
}
RAYCAST_SYSTEM_TEMPLATE_HEADER
typename RS_TEMPL::Snapshot* const RS_TEMPL::GetCurrentSnapshot() const
{
	return &current;
}
RAYCAST_SYSTEM_TEMPLATE_HEADER
typename RS_TEMPL::Snapshot RS_TEMPL::GetCurrentSnapshot()
{
	return &current;
}
RAYCAST_SYSTEM_TEMPLATE_HEADER
typename RS_TEMPL::Snapshot* const RS_TEMPL::GetSnapshotAt(size_t position) const
{
	if (position < snapshots.size())
		return &snapshots[position];
	return nullptr;
}
RAYCAST_SYSTEM_TEMPLATE_HEADER
typename RS_TEMPL::Snapshot* RS_TEMPL::GetSnapshotAt(size_t position)
{
	if (position < snapshots.size())
		return &snapshots[position];
	return nullptr;
}

RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::Subscribe_OnSolution(std::function<void(int, typename RS_TEMPL::Snapshot*)> on_solution)
{
	solutionEvent = on_solution;
	bTriggerOnSolutionEvent = true;
}
RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::Unsubscribe_OnSolution()
{
	solutionEvent = std::function<void(int, typename RS_TEMPL::Snapshot*)>();
	bTriggerOnSolutionEvent = false;
}

// Misc
// ==========================================
RAYCAST_SYSTEM_TEMPLATE_HEADER
void RS_TEMPL::Verbose(const std::string& filename, int skipping)
{
	std::ofstream ofs(filename);
	size_t N = snapshots.size();
	for (int i = 0; i < N; i += skipping)
	{
		auto snapshot = snapshots[i];
		for (auto& ray : snapshot.rayData)
		{
			for (int j = 0; j < SubT::RowsAtCompileTime; ++j)
			ofs << ray.x()[j] << '\t';
			ofs << endl;
		}
		ofs << endl << endl;
	}
}