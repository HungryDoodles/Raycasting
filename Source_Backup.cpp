#include "Header.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <type_traits>

#include "RaycastWR.h"
#include "FiniteDifferenceSolver.h"
#include "ConfigurationManager.h"
#include "ScalarGrid.h"

using namespace std;
using namespace Eigen;

typedef Ray2D<Float> Ray2DT;
typedef Ray2DT::SubT Vector2;
typedef Ray3D<Float> Ray3DT;
typedef Ray3DT::SubT Vector3;

ConfigurationManager config;
FiniteDifferenceSolver<Float> FDM;

string solverType = "FiniteDifference";
uint writeSkip = 25;

uint rayNum = 1000;
Vector2 sphereOrigin = { 0.25, 0.25 };
Float sphereRadius = Float(0.1);
Float deltaTime = 0.005;
Float simulationTime = 1.0;
Float maxDistanceParting = 0.01;
Float collapseDistance = 0.05;
Float maxSubstepping = 0;
Float amplitudePenalty = 0.05;

string integratorType = "GaussLegendre";
bool bCastAsSphere = false; // Cast as plane otherwise
bool bInsertRays = true;
bool bAdaptiveTimestep = false;
bool bLoadGrid = false;
string gridInterpolation = "Linear";

Float FDMdeltaTime = 0.005;
uint FDMsizeX = 100;
uint FDMsizeY = 100;
uint PMLSize = 30;
Float FDMsolTime = 1.0;

string scalarGridName = "xvpcut3.bin";
//string scalarGridDensityName = "rho.bin";
uint scalarGridX = 1401;
uint scalarGridY = 700;
uint smoothingSteps = 30;
Float scalarGridStepX = 5;
Float scalarGridStepY = 5;

void Configuire() 
{
	if (!config.CheckOrOpen("config.ini")) // Proof
	{
		cerr << "Failed to open or parse config.ini" << endl;
		exit(1);
	}

	// Global
	solverType = config.GetOrDefault<string>("config.ini", "Global", "Solver Type (FiniteDifference/Asymptotic/Asymptotic3D)", "FiniteDifference");
	if (solverType != "Asymptotic" && solverType != "FiniteDifference" && solverType != "Asymptotic3D" && solverType != "ScalarGridTest")
	{
		config.Set<string>("config.ini", "Global", "Solver Type (FiniteDifference/Asymptotic/Asymptotic3D)", "FiniteDifference");
		solverType = "FiniteDifference";
	}
	writeSkip = config.GetOrDefault<uint>("config.ini", "Global", "Write skip", 25);

	d_h = config.GetOrDefault("config.ini", "Global", "Numerical step", 1e-6);

	// Raycasting
	rayNum = config.GetOrDefault("config.ini", "Raycasting configuration", "Number of rays", 1000u);
	bInsertRays = config.GetOrDefault("config.ini", "Raycasting configuration", "Insert rays", true);
	bAdaptiveTimestep = config.GetOrDefault("config.ini", "Raycasting configuration", "Adaptive time step", false);
	bLoadGrid = config.GetOrDefault("config.ini", "Raycasting configuration", "Load grid", false);
	deltaTime = config.GetOrDefault("config.ini", "Raycasting configuration", "Time step", 0.005);
	simulationTime = config.GetOrDefault("config.ini", "Raycasting configuration", "Simulation time", 1.0);
	maxDistanceParting = config.GetOrDefault("config.ini", "Raycasting configuration", "Max distance parting", 0.01);
	collapseDistance = config.GetOrDefault("config.ini", "Raycasting configuration", "Collapse distance", 0.025);
	maxSubstepping = config.GetOrDefault("config.ini", "Raycasting configuration", "Max substepping", 0u);
	amplitudePenalty = config.GetOrDefault("config.ini", "Raycasting configuration", "Amplitude penalty", 0.05);

	gridInterpolation = config.GetOrDefault<string>("config.ini", "Raycasting configuration", "Grid Interpolation Type (Linear/Cubic/Hermite/BSpline)", "Linear");
	if (gridInterpolation != "Linear" && gridInterpolation != "Cubic" && gridInterpolation != "Hermite" && gridInterpolation != "BSpline")
	{
		config.Set<string>("config.ini", "Raycasting configuration", "Grid Interpolation Type (Linear/Cubic/Hermite/BSpline)", "Linear");
		gridInterpolation = "Linear";
	}

	string castType = config.GetOrDefault<string>("config.ini", "Raycasting configuration", "Cast type (sphere/plane)", "sphere");

	if (castType != "sphere" && castType != "plane")
	{
		config.Set<string>("config.ini", "Raycasting configuration", "Cast type (sphere/plane)", "sphere");
		castType = "plane";
	}

	integratorType = config.GetOrDefault<string>("config.ini", "Raycasting configuration", "Integrator type (Euler/RK4/GaussLegendre)", "GaussLegendre");

	if (integratorType != "Euler" && integratorType != "RK4" && integratorType != "GaussLegendre")
	{
		config.Set<string>("config.ini", "Raycasting configuration", "Integrator type (Euler/RK4/GaussLegendre)", "GaussLegendre");
		integratorType = "GaussLegendre";
	}

	sphereOrigin.x() = config.GetOrDefault("config.ini", "Raycasting configuration", "Sphere X", Float(0.25));
	sphereOrigin.y() = config.GetOrDefault("config.ini", "Raycasting configuration", "Sphere Y", Float(0.25));
	sphereRadius = config.GetOrDefault("config.ini", "Raycasting configuration", "Sphere Radius", Float(0.1));

	// FDM
	FDMdeltaTime = config.GetOrDefault("config.ini", "FDM configuration", "Time step", Float(0.005));
	FDMsizeX = config.GetOrDefault("config.ini", "FDM configuration", "Size X", 100u);
	FDMsizeY = config.GetOrDefault("config.ini", "FDM configuration", "Size Y", 100u);
	PMLSize = config.GetOrDefault("config.ini", "FDM configuration", "PML Size", 30u);
	FDMsolTime = config.GetOrDefault("config.ini", "FDM configuration", "Simulation time", Float(1.0));

	// Scalar grid
	scalarGridName = config.GetOrDefault<string>("config.ini", "Scalar Grid", "Name", "x - vp - cut - 3.bin");
	scalarGridX = config.GetOrDefault("config.ini", "Scalar Grid", "Grid size X", 1401u);
	scalarGridY = config.GetOrDefault("config.ini", "Scalar Grid", "Grid size Y", 700u);
	scalarGridStepX = config.GetOrDefault("config.ini", "Scalar Grid", "Grid step X", Float(5));
	scalarGridStepY = config.GetOrDefault("config.ini", "Scalar Grid", "Grid step Y", Float(5));
	smoothingSteps = config.GetOrDefault("config.ini", "Scalar Grid", "Smoothing steps", 30u);
}

template<class RaycastSystemType>
void RunTest() 
{
	RaycastSystemType rs;

	rs.bInsertRays = bInsertRays;
	rs.bEvaluateDeltaTime = bAdaptiveTimestep;
	rs.max_distance_parting = maxDistanceParting;
	rs.collapse_distance = collapseDistance;
	rs.bEnclosedShape = bCastAsSphere;
	rs.SetDeltaTime(::deltaTime);
	rs.maxSubstepping = maxSubstepping;
	rs.smoothingSteps = smoothingSteps;
	rs.amplitude_penalty = amplitudePenalty;
	if (bLoadGrid) if (!rs.LoadScalarGrid(scalarGridName, scalarGridX, scalarGridY, Vector2{ scalarGridStepX, scalarGridStepY }))
		cout << "Failed to load scalar grid" << endl;
	if (gridInterpolation == "Linear")
		rs.samplerType = ScalarGridSamplerType::Linear;
	if (gridInterpolation == "Cubic")
		rs.samplerType = ScalarGridSamplerType::Cubic;
	if (gridInterpolation == "Hermite")
		rs.samplerType = ScalarGridSamplerType::Hermite;
	if (gridInterpolation == "BSpline")
		rs.samplerType = ScalarGridSamplerType::BSpline;

	if (bCastAsSphere)
	{
		rs.Create(rayNum, [](uint n, void* rs_in)->Ray2DT
			{
				RaycastSystemType* rs = (RaycastSystemType*)rs_in;
				double angle = (M_PI * 2. * n) / rayNum;
				Vector2 normal = Rotation2D<Float>(Float(angle)) * Vector2 { 1, 0 };
				Ray2DT ray;
				ray.x(normal * sphereRadius + sphereOrigin);
				ray.p(normal / rs->C(ray.x()));
				return ray;
			});
	}
	else
	{
		rs.Create(rayNum, [](uint n, void* rs_in)->Ray2DT
			{
				RaycastSystemType* rs = (RaycastSystemType*)rs_in;
				Ray2DT ray;
				if (bLoadGrid)
				{
					ray.x({ Float(n) / rayNum * scalarGridX * scalarGridStepX, scalarGridY * scalarGridStepY * (1.0 - 0.1) });
					ray.p(Vector2({ 0, -1 }) / rs->C(ray.x()));
				}
				else
				{
					ray.x({ Float(n) / rayNum, 1 });
					ray.p(Vector2({ 0, -1 }) / rs->C(ray.x()));
				}
				return ray;
			});
	}

	ofstream outf("output.txt");
	ScalarGrid<Float> ray_sg;
	ray_sg.Resize(scalarGridX / 2, scalarGridY / 2);
	ray_sg.BeginGif("output_anim.gif", writeSkip);
	rs.Subscribe_OnSolution([&outf, &ray_sg, &rs](int step, typename RaycastSystemType::Snapshot* snapshot)->void
	{
		if ((step - 1) % writeSkip == 0)
		{
			ray_sg.Zero();
			rs.ComputeAmplitudes();
			for (int i = 0; i < snapshot->rayData.size(); ++i)
			{
				auto& ray = snapshot->rayData[i];
				for (int j = 0; j < RaycastSystemType::SubT::RowsAtCompileTime; ++j)
					outf << ray.x()[j] << '\t';
				outf << endl;

				ray_sg.Project_Bilinear(ray.x()[0] / scalarGridStepX * Float(0.5), ray.x()[1] / scalarGridStepY * Float(0.5), snapshot->payload.ampData[i].computed_amplitude);
			}
			outf << endl << endl;
			ray_sg.ExpKernel(); ray_sg.ExpKernel(); ray_sg.ExpKernel();
			ray_sg.WriteGifFrame(1.0, 0.0);
		}
	});

	rs.SimulateAsync(simulationTime);

	stringstream messageSS;
	messageSS.precision(3);
	while (rs.asyncStatus.bWorking)
	{
		cout << '\r';
		messageSS = stringstream();
		messageSS << "T: " << rs.asyncStatus.current_time << '\t'
			<< "Step:" << rs.asyncStatus.steps_num << '\t'
			<< "Rays:" << rs.asyncStatus.rays_num << "            ";

		cout << messageSS.str();

		this_thread::sleep_for(16ms);
	}
	cout << endl;
	ray_sg.EndGif();
	
	if (is_same<RaycastSystemType, RaycastWR<Float, Ray2D, IntegratorGaussLegendre>>::value)
	{
		using IntegratorType = IntegratorGaussLegendre<Float, Ray2D<Float>::VecT>;

		auto& iters_hits = IntegratorType::iterations_limit_stops;
		auto& prec_hits = IntegratorType::precision_limit_stops;
		auto total_hits = iters_hits + prec_hits;
		auto& total_iterations = IntegratorType::total_iterations;
		auto& total_calls = IntegratorType::total_calls;
		auto& max_error = IntegratorType::max_error;

		cout << "Gauss-Legendre stats:" << endl;
		cout << "Total number of iteration limit been reached: " << iters_hits << "\t (" << (Float(iters_hits) / total_hits * 100) << "%)" << endl;
		cout << "Total number of precision limit been reached: " << prec_hits << "\t (" << (Float(prec_hits) / total_hits * 100) << "%)" << endl;
		cout << "Average number of iterations taken: " << (Float(total_iterations) / total_calls) << endl;
		cout << "Max error: " << max_error << endl;
	}

	//rs.Verbose("output.txt");
}

template<class RaycastSystemType>
void RunTest3D()
{
	RaycastSystemType rs;

	rs.bInsertRays = false;
	if (bInsertRays) cout << "Ray instertion is not yet supported in 3D mode" << endl;
	rs.max_distance_parting = maxDistanceParting;
	rs.bEnclosedShape = bCastAsSphere;
	rs.SetDeltaTime(::deltaTime);
	rs.maxSubstepping = maxSubstepping;
	if (gridInterpolation == "Linear")
		rs.samplerType = ScalarGridSamplerType::Linear;
	if (gridInterpolation == "Cubic")
		rs.samplerType = ScalarGridSamplerType::Cubic;
	if (gridInterpolation == "Hermite")
		rs.samplerType = ScalarGridSamplerType::Hermite;
	if (gridInterpolation == "BSpline")
		rs.samplerType = ScalarGridSamplerType::BSpline;

	if (bCastAsSphere)
	{
		cout << "Sphere cast in 3D is not supported" << endl;
	}
	{
		rs.Create(rayNum * rayNum, [](uint n, void* rs_in)->Ray3DT
			{
				RaycastSystemType* rs = (RaycastSystemType*)rs_in;
				Ray3DT ray;
				int ix = int(n) % rayNum;
				int iy = int(n) / rayNum;
				ray.x({ Float(ix) / rayNum, Float(iy) / rayNum, Float(1.0) });
				ray.p(Vector3{ 0, 0, -1 } / rs->C(ray.x()));
				return ray;
			});
	}

	ofstream outf("output3D.txt");
	rs.Subscribe_OnSolution([&outf](int step, typename RaycastSystemType::Snapshot* snapshot)->void
	{
		if (step % writeSkip == 0)
		{
			for (auto& ray : snapshot->rayData)
			{
				for (int j = 0; j < RaycastSystemType::SubT::RowsAtCompileTime; ++j)
					outf << ray.x()[j] << '\t';
				outf << endl;
			}
			outf << endl << endl;
		}
	});

	rs.SimulateAsync(simulationTime);

	stringstream messageSS;
	messageSS.precision(3);
	while (rs.asyncStatus.bWorking)
	{
		cout << '\r';
		messageSS = stringstream();
		messageSS << "T: " << rs.asyncStatus.current_time << '\t'
			<< "Step:" << rs.asyncStatus.steps_num << '\t'
			<< "Rays:" << rs.asyncStatus.rays_num << "            ";

		cout << messageSS.str();

		this_thread::sleep_for(16ms);
	}
	cout << endl;

	if (is_same<RaycastSystemType, RaycastWR<Float, Ray2D, IntegratorGaussLegendre>>::value)
	{
		using IntegratorType = IntegratorGaussLegendre<Float, Ray2D<Float>::VecT>;

		auto& iters_hits = IntegratorType::iterations_limit_stops;
		auto& prec_hits = IntegratorType::precision_limit_stops;
		auto total_hits = iters_hits + prec_hits;
		auto& total_iterations = IntegratorType::total_iterations;
		auto& total_calls = IntegratorType::total_calls;
		auto& max_error = IntegratorType::max_error;

		cout << "Gauss-Legendre stats:" << endl;
		cout << "Total number of iteration limit been reached: " << iters_hits << "\t (" << (Float(iters_hits) / total_hits * 100) << "%)" << endl;
		cout << "Total number of precision limit been reached: " << prec_hits << "\t (" << (Float(prec_hits) / total_hits * 100) << "%)" << endl;
		cout << "Average number of iterations taken: " << (Float(total_iterations) / total_calls) << endl;
		cout << "Max error: " << max_error << endl;
	}

	//rs.Verbose("output3D.txt", 2);
}

void RunFDM() 
{
	FDM.SetPMLSize(PMLSize);
	FDM.Resize({ FDMsizeX, FDMsizeY });
	FDM.SetDim({ 1.0, 1.0 });

	constexpr Float sigma = 0.006 * 6;
	Float sqrtsqrt_pi = sqrt(sqrt(M_PI));

	FDM.AddSolutionInHistory([sqrtsqrt_pi](uint x, uint y)->Float
		{ 
			Float fy = Float(y) / FDMsizeY;
			Float t = (fy - 0.2 - FDMdeltaTime * 0);
			//return ((x != 0 && x != (FDMsizeX - 1)) ? 
			//	(2.0 / (sqrt(3 * sigma) * sqrtsqrt_pi)) * (1 - (t * t / (sigma * sigma))) * exp(-(t * t) / (2 * sigma * sigma)) : 0.0); 
			return (2.0 / (sqrt(3 * sigma) * sqrtsqrt_pi)) * (1 - (t * t / (sigma * sigma))) * exp(-(t * t) / (2 * sigma * sigma));
		});
	FDM.AddSolutionInHistory([sqrtsqrt_pi](uint x, uint y)->Float
		{ 
			Float fy = Float(y) / FDMsizeY; 
			Float t = (fy - 0.2 - FDMdeltaTime * 1);
			//return ((x != 0 && x != (FDMsizeX - 1)) ? 
			//	(2.0 / (sqrt(3 * sigma) * sqrtsqrt_pi)) * (1 - (t * t / (sigma * sigma))) * exp(-(t * t) / (2 * sigma * sigma)) : 0.0);
			return (2.0 / (sqrt(3 * sigma) * sqrtsqrt_pi)) * (1 - (t * t / (sigma * sigma))) * exp(-(t * t) / (2 * sigma * sigma));
		});

	FDM.FillVelocityGrid([](uint x, uint y)->Float 
		{
			Float xf = Float(x) / (FDMsizeX - 1);
			Float yf = Float(y) / (FDMsizeY - 1);

			Vector2 x0 = { 0.5, 0.5 };
			Float A = Float(1.0);
			Float B = Float(0.25);
			Float C = Float(-100);

			return A - B * exp((x0 - Vector2{ xf, yf }).squaredNorm() * C);
		});

	FDM.SetTimeStep(FDMdeltaTime);

	if (!FDM.InitOpenCL()) 
	{
		cerr << "OpenCL init failed" << endl;
		return;
	}
	if (!FDM.SetBuffers())
	{
		cerr << "OpenCL buffers init failed" << endl;
		return;
	}

	ScalarGrid<Float> sg;
	sg.Resize(FDMsizeX + 2 * PMLSize, FDMsizeY + 2 * PMLSize);
	sg.BeginGif("FDM_anim.gif", 1);

	FDM.Subscribe_OnSolution(
		// Decide
		[](int index)->bool { return index % writeSkip == 0; },
		// Write
		[&sg](int index, decltype(FDM)::VectorX* sol)->void
		{
			memcpy(sg.Data(), (*sol).data(), sizeof(Float) * (FDMsizeX + 2 * PMLSize) * (FDMsizeY + 2 * PMLSize));
			sg.WriteGifFrame(-0.25, 0.25,
				10,132,200,
				255,132,66);
		});

	FDM.Solve(FDMsolTime);

	sg.EndGif();
	FDM.Clear();
}

void ScalarGridTest() 
{
	cout << "Testing scalar grid" << endl;

	ScalarGrid<Float> sg;
	if (!sg.ReadFromBinary(scalarGridName, scalarGridX, scalarGridY)) 
	{
		cerr << "Failed to load " << scalarGridName << endl;
	}
	ofstream outf("ScalarGridTest.txt");
	ofstream outfd0("ScalarGridDif0Test.txt");
	ofstream outfd1("ScalarGridDif1Test.txt");

	ScalarGridSamplerType sampler = ScalarGridSamplerType::BSpline;

	for (int i = 0; i < 20; ++i) sg.ExpKernel();
	outf.precision(13);
	for (int i = 0; i < 25000; ++i)
	{
		Float x = (0 * Float(i) / 100 + 50);
		Float y = Float(i) / 400 + 290;
		outf << sg.Sample(sampler, x, y) << endl;
		outfd0 << sg.SampleDerivative(sampler, 0, x, y) << endl;
		outfd1 << sg.SampleDerivative(sampler, 1, x, y) << endl;
	}
}

void BSplineTest() 
{
	cout << "Testing BSplines" << endl;

	ofstream outf("BSplineTest.txt");
	ofstream outf_der("BSplineTestDeriv.txt");

	int p = 7;
	std::vector<Float> y { 0, 0, 0, 1, 3, 2, 2, 2, 2, 2, 2, 2 };
	auto poke = [](int index, const std::vector<Float>& v)->Float 
	{ 
		if (index < 0 || index >= v.size()) 
			return 0;
		return v[index];
	};

	for (Float x = 0; x < 12; x += 0.02)
	{
		int ix = (int)x;
		Float fx = x - ix;
		static std::vector<Float> c;
		c.resize(1 + p);
		for (int i = ix; i < ix + c.size(); ++i)
			c[i - ix] = poke(i - p / 2, y);
		Float val = BSplineDeBoor(fx, p, c);
		outf << x << '\t' << val << endl;
		Float der = (BSplineDeBoor(fx + d_h, p, c) - BSplineDeBoor(fx - d_h, p, c)) / (2 * d_h);
		Float der2 = (BSplineDeBoor(fx + d_h, p, c) - 2 * BSplineDeBoor(fx, p, c) + BSplineDeBoor(fx - d_h, p, c)) / (2 * d_h);
		outf_der << x << '\t' << der << '\t' << der2 << endl;
	}
}

int main() 
{
	FiniteDifferenceSolver<Float>::VectorX vect;

	cout << "Floating point type size: " << sizeof(Float) << endl;

	Configuire();

	if (solverType == "FiniteDifference") 
	{
		RunFDM();
	}
	if (solverType == "Asymptotic")
	{
		if (integratorType == "Euler")
		{
			RunTest<RaycastWR<Float, Ray2D, IntegratorEuler>>();
		}
		if (integratorType == "RK4")
		{
			RunTest<RaycastWR<Float, Ray2D, IntegratorRK4Expl>>();
		}
		if (integratorType == "GaussLegendre")
		{
			RunTest<RaycastWR<Float, Ray2D, IntegratorGaussLegendre>>();
		}
	}
	if (solverType == "Asymptotic3D") 
	{
		if (integratorType == "Euler")
		{
			RunTest3D<RaycastWR<Float, Ray3D, IntegratorEuler>>();
		}
		if (integratorType == "RK4")
		{
			RunTest3D<RaycastWR<Float, Ray3D, IntegratorRK4Expl>>();
		}
		if (integratorType == "GaussLegendre")
		{
			RunTest3D<RaycastWR<Float, Ray3D, IntegratorGaussLegendre>>();
		}
	}
	if (solverType == "ScalarGridTest") 
	{
		ScalarGridTest();
		//BSplineTest();
	}

	cout << "Press enter to continue" << endl;
	//cin.get();
	return 0;
}