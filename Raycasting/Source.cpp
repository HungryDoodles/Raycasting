#include "Header.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <queue>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <type_traits>
#include <chrono>

#include "ConfigurationManager.h"
#include "RaycastWR.h"
#include "FiniteDifferenceSolver.h"
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
Float startOffset = 0.1;
bool bForceCPU = true;
uint gifDelay = 10;

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
bool bFastDiff = true;
string gridInterpolation = "Linear";

Float FDMdeltaTime = 0.005;
uint FDMsizeX = 100;
uint FDMsizeY = 100;
uint PMLSize = 30;
Float FDMsolTime = 1.0;
Float FDMSigma = 0.006;
bool bUseScalarGrid = false;

string scalarGridName = "xvpcut3.bin";
//string scalarGridDensityName = "rho.bin";
uint scalarGridX = 1401;
uint scalarGridY = 700;
uint smoothingSteps = 30;
Float scalarGridStepX = 5;
Float scalarGridStepY = 5;


float vector_poke(int i, const vector<float>& v)
{
	if (v.size() == 0)
		return 0;
	if (i < 0)
		return v.front();
	if (i >= v.size())
		return v.back();
	return v[i];
}
uint32_t ShinyGradientLaw(float value, float min, float max)
{
	// Define points
	const vector<float> r{ 0.0, 0.3, 0.8, 0.9, 1.0 };
	const vector<float> g{ 0.0, 0.0, 0.1, 0.5, 0.9 };
	const vector<float> b{ 0.0, 0.6, 0.1, 0.0, 0.0 };
	static thread_local vector<float> r_cp, g_cp, b_cp;

	double width = max - min;
	double alpha = (value - min) / width * r.size();

	int ix = (int)floor(alpha);
	double x = alpha - ix;

	const int order = 3;

	r_cp.resize(order + 1);
	g_cp.resize(order + 1);
	b_cp.resize(order + 1);

	for (int i = 0; i < order + 1; ++i)
	{
		r_cp[i] = vector_poke(i + ix - order / 2, r);
		g_cp[i] = vector_poke(i + ix - order / 2, g);
		b_cp[i] = vector_poke(i + ix - order / 2, b);
	}

	float r_f = BSplineDeBoor<float>(x, order, r_cp);
	float g_f = BSplineDeBoor<float>(x, order, g_cp);
	float b_f = BSplineDeBoor<float>(x, order, b_cp);

	uint8_t r8 = (uint8_t)Lerp<float>(r_f, 0, 255);
	uint8_t g8 = (uint8_t)Lerp<float>(g_f, 0, 255);
	uint8_t b8 = (uint8_t)Lerp<float>(b_f, 0, 255);

	return 0xff000000 | (r8 << 0) | (g8 << 8) | (b8 << 16);
}
uint32_t TransparentBWLaw(float value, float min, float max, uint32_t baseColor = 0xffffffff) 
{
	const vector<float> v{ 0.0, 1.0 };
	static thread_local vector<float> v_cp;

	double width = max - min;
	double alpha = (value - min) / width * v.size();

	int ix = (int)floor(alpha);
	double x = alpha - ix;

	const int order = 3;

	v_cp.resize(order + 1);

	for (int i = 0; i < order + 1; ++i)
		v_cp[i] = vector_poke(i + ix - order / 2, v);

	float v_f = BSplineDeBoor<float>(x, order, v_cp);
	uint8_t v8 = (uint8_t)Lerp<float>(v_f, 0, 255);

	return (v8 << 24) | (baseColor & 0x00ffffff);
}
Float RickerWavelet(Float d2) 
{
	//return (2.0 / (sqrt(3 * sigma) * sqrtsqrt_pi)) * (1 - (t * t / (sigma * sigma))) * exp(-(t * t) / (2 * sigma * sigma));
	return 1.0 / (M_PI * pow(FDMSigma, 4)) * (1 - 0.5 * d2 / (FDMSigma * FDMSigma) ) * exp(-d2 / (2 * FDMSigma * FDMSigma));
}
void Dump(ScalarGrid<Float>& sg, const string filename = "Default_dump.gif", Float min_v = 0, Float max_v = 1)
{
	sg.BeginGif(filename, 0);
	sg.WriteGifFrame([min_v, max_v](Float value)->uint32_t { return ShinyGradientLaw(value, min_v, max_v); });
	sg.EndGif();
}

void Configuire() 
{
	if (!config.CheckOrOpen("config.ini")) // Proof
	{
		cerr << "Failed to open or parse config.ini" << endl;
		exit(1);
	}

	// Global
	solverType = config.GetOrDefault<string>("config.ini", "Global", "Solver Type (FiniteDifference/Asymptotic/Asymptotic3D/Hybrid)", "FiniteDifference");
	if (solverType != "Asymptotic" && solverType != "FiniteDifference" && solverType != "Asymptotic3D" && solverType != "ScalarGridTest" && solverType != "Hybrid")
	{
		config.Set<string>("config.ini", "Global", "Solver Type (FiniteDifference/Asymptotic/Asymptotic3D/Hybrid)", "FiniteDifference");
		solverType = "FiniteDifference";
	}
	writeSkip = config.GetOrDefault<uint>("config.ini", "Global", "Write skip", 25);
	startOffset = config.GetOrDefault<Float>("config.ini", "Global", "Start offset", Float(0.1));
	bForceCPU = config.GetOrDefault<bool>("config.ini", "Global", "Force CPU", true);
	gifDelay = config.GetOrDefault<uint>("config.ini", "Global", "GIF delay", 10);

	d_h = config.GetOrDefault("config.ini", "Global", "Numerical step", 1e-6);

	// Raycasting
	rayNum = config.GetOrDefault("config.ini", "Raycasting configuration", "Number of rays", 1000u);
	bInsertRays = config.GetOrDefault("config.ini", "Raycasting configuration", "Insert rays", true);
	bAdaptiveTimestep = config.GetOrDefault("config.ini", "Raycasting configuration", "Adaptive time step", false);
	bLoadGrid = config.GetOrDefault("config.ini", "Raycasting configuration", "Load grid", false);
	bFastDiff = config.GetOrDefault("config.ini", "Raycasting configuration", "Fast differential", true);
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
	FDMSigma = config.GetOrDefault("config.ini", "FDM configuration", "Wavelet sigma", Float(1.0));
	bUseScalarGrid = config.GetOrDefault("config.ini", "FDM configuration", "Use scalar grid", false);

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
	rs.bFastDifferential = bFastDiff;
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
					Float alpha = Float(n) / rayNum * (1.0 - 2.0 * startOffset) + startOffset;
					ray.x(
						{
							scalarGridX * scalarGridStepX * alpha,
							scalarGridY * scalarGridStepY * (1.0 - startOffset)
						});
					ray.p(Vector2({ 0, -1 }) / rs->C(ray.x()));
				}
				else
				{
					ray.x({ Float(n) / rayNum, 1 });
					ray.p(Vector2({ 0, -1 }) / rs->C(ray.x()));
				}
				return ray;
			});
		/*rs.Create(rayNum, [](uint n, void* rs_in)->Ray2DT
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
			});*/
	}

	ofstream outf("output.txt");
	ScalarGrid<Float> ray_sg, vel_sg;
	if (bLoadGrid)
		vel_sg.ReadFromBinary(scalarGridName, scalarGridX, scalarGridY);
	else
		vel_sg.Zero();
	ray_sg.Resize(scalarGridX, scalarGridY);
	vel_sg.FlipY();
	vel_sg.BeginGif("output_anim.gif", gifDelay);

	vector<function<uint32_t(float)>> laws
	{
		[](float v)->uint32_t { return ShinyGradientLaw(v, 1500, 5500); },
		[](float v)->uint32_t { return TransparentBWLaw(v, 0, 1); }
	};

	chrono::duration<long long, std::nano> gifWriteDuration(0);
	rs.Subscribe_OnSolution([&](int step, typename RaycastSystemType::Snapshot* snapshot)->void
	{
		if ((step - 1) % writeSkip == 0)
		{
			auto beginWritePoint = chrono::steady_clock::now();
			ray_sg.Zero();
			rs.ComputeAmplitudes();
			for (int i = 0; i < snapshot->rayData.size(); ++i)
			{
				auto& ray = snapshot->rayData[i];
				for (int j = 0; j < RaycastSystemType::SubT::RowsAtCompileTime; ++j)
					outf << ray.x()[j] << '\t';
				outf << endl;

				//ray_sg.Project_Bilinear(ray.x()[0] / scalarGridStepX * Float(0.5), ray.x()[1] / scalarGridStepY * Float(0.5), snapshot->payload.ampData[i].computed_amplitude);
				/*ray_sg.Project_Bilinear(
					ray.x()[0] / scalarGridStepX, 
					ray.x()[1] / scalarGridStepY, 
					4.0);*/

				ray_sg.Project_Bilinear(ray.x()[0] / scalarGridStepX - 0.5, ray.x()[1] / scalarGridStepY - 0.5, 4.0);
				ray_sg.Project_Bilinear(ray.x()[0] / scalarGridStepX - 0.5, ray.x()[1] / scalarGridStepY + 0.5, 4.0);
				ray_sg.Project_Bilinear(ray.x()[0] / scalarGridStepX + 0.5, ray.x()[1] / scalarGridStepY - 0.5, 4.0);
				ray_sg.Project_Bilinear(ray.x()[0] / scalarGridStepX + 0.5, ray.x()[1] / scalarGridStepY + 0.5, 4.0);
			}
			outf << endl << endl;

			//ray_sg.ExpKernel(); ray_sg.ExpKernel(); ray_sg.ExpKernel();
			//ray_sg.WriteGifFrame(1.0, 0.0);
			ray_sg.FlipY();
			vel_sg.WriteGifFrameLayered(laws.data(), &ray_sg, 1);
			gifWriteDuration += chrono::steady_clock::now() - beginWritePoint;
		}
	});

	rs.SimulateAsync(simulationTime);
	chrono::time_point startTimePoint = chrono::steady_clock::now();

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

		this_thread::sleep_for(30ms);
	}
	auto duration = (chrono::steady_clock::now() - startTimePoint) - gifWriteDuration;
	int seconds = chrono::duration_cast<chrono::seconds>(duration).count();
	int milliseconds = chrono::duration_cast<chrono::milliseconds>(duration).count() - seconds * 1000;
	cout << endl;
	cout << "Run time (excluding write): " << seconds << "s" << milliseconds << "m +-60ms" << endl;
	vel_sg.EndGif();
	
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
	rs.bFastDifferential = bFastDiff;
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
	FDM.bForceCPU = bForceCPU;

	Float RickerZero = RickerWavelet(0);

	FDM.AddSolutionInHistory([RickerZero](uint x, uint y)->Float
		{ 
			Float fx = Float(x) / FDMsizeX;
			Float fy = Float(y) / FDMsizeY;
			Float d2 = 0;
			
			if (fx < startOffset)
				d2 = (startOffset - fx) * (startOffset - fx) + (startOffset - fy) * (startOffset - fy);
			else if (fx > 1.0 - startOffset)
				d2 = (1.0 - startOffset - fx) * (1.0 - startOffset - fx) + (startOffset - fy) * (startOffset - fy);
			else
				d2 = (startOffset - fy) * (startOffset - fy);

			return RickerWavelet(d2) / RickerZero;
		});
	FDM.AddSolutionInHistory([RickerZero](uint x, uint y)->Float
		{ 
			Float fx = Float(x) / FDMsizeX;
			Float fy = (Float(y)) / FDMsizeY - FDMdeltaTime * 1.0;
			Float d2 = 0;

			if (fx < startOffset)
				d2 = (startOffset - fx) * (startOffset - fx) + (startOffset - fy) * (startOffset - fy);
			else if (fx > 1.0 - startOffset)
				d2 = (1.0 - startOffset - fx) * (1.0 - startOffset - fx) + (startOffset - fy) * (startOffset - fy);
			else
				d2 = (startOffset - fy) * (startOffset - fy);

			return RickerWavelet(d2) / RickerZero;
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
	sg.BeginGif("FDM_anim.gif", gifDelay);

	FDM.Subscribe_OnSolution(
		// Decide
		[](int index)->bool { return index % writeSkip == 0; },
		// Write
		[&sg](int index, decltype(FDM)::VectorX* sol)->void
		{
			memcpy(sg.Data(), (*sol).data(), sizeof(Float) * (FDMsizeX + 2 * PMLSize) * (FDMsizeY + 2 * PMLSize));
			sg.WriteGifFrame([](double v)->double { return ShinyGradientLaw(v, -1, 1); });
			/*sg.WriteGifFrame(-0.25, 0.25,
				10,132,200,
				255,132,66);*/
		});

	FDM.Solve(FDMsolTime);

	sg.EndGif();
	FDM.Clear();
}

void RunFDM_Marm() 
{
	FDMsizeX = scalarGridX;
	FDMsizeY = scalarGridY;

	ScalarGrid<Float> velGrid;
	velGrid.ReadFromBinary(scalarGridName, scalarGridX, scalarGridY, 1, false);

	FDM.SetPMLSize(PMLSize);
	FDM.LoadVelocityGrid(velGrid);
	FDM.SetDim({ scalarGridX * scalarGridStepX, scalarGridY * scalarGridStepY });
	FDM.bForceCPU = bForceCPU;

	Float RickerZero = RickerWavelet(0);

	FDM.AddSolutionInHistory([RickerZero](uint x, uint y)->Float
		{
			Float fx = Float(x) / FDMsizeX;
			Float fy = Float(y) / FDMsizeY;
			Float d2 = 0;

			if (fx < startOffset)
				d2 = (startOffset - fx) * (startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else if (fx > 1.0 - startOffset)
				d2 = (1.0 - startOffset - fx) * (1.0 - startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else
				d2 = (startOffset - fy) * (startOffset - fy) * FDMsizeY;

			return RickerWavelet(d2) / RickerZero;
		});
	FDM.AddSolutionInHistory([RickerZero, &velGrid](uint x, uint y)->Float
		{
			Float fx = Float(x) / FDMsizeX;
			Float fy = Float(y - FDMdeltaTime * velGrid.At(x, y) / scalarGridStepY) / FDMsizeY;
			Float d2 = 0;

			if (fx < startOffset)
				d2 = (startOffset - fx) * (startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else if (fx > 1.0 - startOffset)
				d2 = (1.0 - startOffset - fx) * (1.0 - startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else
				d2 = (startOffset - fy) * (startOffset - fy) * FDMsizeY;

			return RickerWavelet(d2) / RickerZero;
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
	sg.BeginGif("FDM_anim.gif", gifDelay);

	FDM.Subscribe_OnSolution(
		// Decide
		[](int index)->bool { return index % writeSkip == 0; },
		// Write
		[&sg](int index, decltype(FDM)::VectorX* sol)->void
		{
			memcpy(sg.Data(), (*sol).data(), sizeof(Float) * (FDMsizeX + 2 * PMLSize) * (FDMsizeY + 2 * PMLSize));
			sg.WriteGifFrame([](double v)->double { return ShinyGradientLaw(v, -1, 1); });
		});

	auto startTimePoint = chrono::steady_clock::now();
	FDM.Solve(FDMsolTime);
	double durationSec = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - startTimePoint).count() * 1e-6;

	cout << "Duration: " << durationSec << " s" << endl;

	sg.EndGif();
	FDM.Clear();
}


struct WriterWorkerData 
{
	bool bRunWriterLoop = false;

	mutex rayDataMutex;
	mutex gridDataMutex;

	queue<void*> raySnapshots;
	queue<FiniteDifferenceSolver<Float>::VectorX> gridSnapshots;

} writerWorkerData;

template<class RaycastSystemType>
void WriterWorker()
{
	typedef typename RaycastSystemType::Snapshot RaySnapshot;
	typedef typename FiniteDifferenceSolver<Float>::VectorX GridSnapshot;
	typedef typename Eigen::Matrix<Float, -1, 1> VectorXf;

	auto& w = writerWorkerData;

	bool bHasSnapshot = false;
	RaySnapshot localRaySnapshot;
	GridSnapshot localGridSnapshot;

	uint gifSizeX = FDMsizeX + 2 * PMLSize;
	uint gifSizeY = FDMsizeY + 2 * PMLSize;

	vector<function<uint32_t(float)>> laws 
	{ 
		[](float v)->uint32_t { return ShinyGradientLaw(v, -1, 1); },
		[](float v)->uint32_t { return TransparentBWLaw(v, 0, 1, 0x00000000); }
	};

	ScalarGrid<Float> gridSG(gifSizeX, gifSizeY), raySG(gifSizeX, gifSizeY);
	vector<VectorXf> artificialSeismogramData;
	gridSG.BeginGif("Hybrid_anim.gif", writeSkip * 28 / 500);

	while (w.bRunWriterLoop)
	{
		{
			lock_guard<mutex> rayMutexGuard(w.rayDataMutex);
			lock_guard<mutex> gridMutexGuard(w.gridDataMutex);

			if (!w.raySnapshots.empty() && !w.gridSnapshots.empty())
			{
				RaySnapshot* rayPtr = (RaySnapshot*)w.raySnapshots.front();
				localRaySnapshot = *rayPtr;
				localGridSnapshot = w.gridSnapshots.front();
				delete rayPtr;
				w.raySnapshots.pop();
				w.gridSnapshots.pop();
				bHasSnapshot = true;
			}
		}

		if (bHasSnapshot) 
		{
			memcpy(gridSG.Data(), localGridSnapshot.data(), sizeof(Float) * (FDMsizeX + 2 * PMLSize) * (FDMsizeY + 2 * PMLSize));
			artificialSeismogramData.push_back(localGridSnapshot.segment((FDMsizeY + PMLSize - 1) * (FDMsizeX + 2 * PMLSize), FDMsizeX)); // Capture seismogram data

			raySG.Zero();
			for (int i = 0; i < localRaySnapshot.rayData.size(); ++i)
			{
				auto& ray = localRaySnapshot.rayData[i];

				raySG.Project_Bilinear(ray.x()[0] / scalarGridStepX + PMLSize - 0.5, ray.x()[1] / scalarGridStepY + PMLSize - 0.5, 4.0);
				raySG.Project_Bilinear(ray.x()[0] / scalarGridStepX + PMLSize - 0.5, ray.x()[1] / scalarGridStepY + PMLSize + 0.5, 4.0);
				raySG.Project_Bilinear(ray.x()[0] / scalarGridStepX + PMLSize + 0.5, ray.x()[1] / scalarGridStepY + PMLSize - 0.5, 4.0);
				raySG.Project_Bilinear(ray.x()[0] / scalarGridStepX + PMLSize + 0.5, ray.x()[1] / scalarGridStepY + PMLSize + 0.5, 4.0);
			}

			// Amen
			gridSG.WriteGifFrameLayered(laws.data(), &raySG, 1);

			bHasSnapshot = false;
		}

		this_thread::sleep_for(1ms);
	}

	constexpr int stretchY = 25;
	ScalarGrid<Float> artificialSeismogram(FDMsizeX, artificialSeismogramData.size() * stretchY);
	artificialSeismogram.Process([&artificialSeismogramData](Float value, int x, int y, int z)->Float 
		{
			float fy = float(y);
			int base = int(fy / stretchY);
			float alpha = fy / stretchY - base;
			int next = std::min<int>(artificialSeismogramData.size() - 1, base + 1);
			return artificialSeismogramData[base][x] * (1 - alpha) + artificialSeismogramData[next][x] * alpha;
		});
	artificialSeismogram.BeginGif("ArtificialSeismogram.gif", 0);
	artificialSeismogram.WriteGifFrame([](float value)->uint32_t {return ShinyGradientLaw(value, -1, 1); });
	artificialSeismogram.EndGif();

	gridSG.EndGif();
}
template<class RaycastSystemType>
void RunHybrid() 
{
	// Get velocity grid
	ScalarGrid<Float> velGrid;

	if (bUseScalarGrid)
	{
		velGrid.ReadFromBinary(scalarGridName, scalarGridX, scalarGridY, 1, false);

	// ==================================
	// Decide FDM grid

		FDMsizeX = scalarGridX;
		FDMsizeY = scalarGridY;
	}
	else 
	{
		scalarGridStepX = 1.0 / (FDMsizeX);
		scalarGridStepY = 1.0 / (FDMsizeY);

		velGrid.Resize(FDMsizeX, FDMsizeY);
		velGrid.Process([](Float v, uint x, uint y, uint z)->Float 
			{
				Float xf = Float(x) / (FDMsizeX - 1);
				Float yf = Float(y) / (FDMsizeY - 1);

				Vector2 x0 = { 0.5, 0.5 };
				Float A = Float(1.0);
				Float B = Float(0.25);
				Float C = Float(-100);

				return A - B * exp((x0 - Vector2{ xf, yf }).squaredNorm() * C);
			});
	}
	if (bUseScalarGrid)
		Dump(velGrid, "VelGrid_Hybrid_Dump.gif", 1500, 5500);
	else
		Dump(velGrid, "VelGrid_Hybrid_Dump.gif", 0.75, 1);

	FDM.SetPMLSize(PMLSize);
	FDM.LoadVelocityGrid(velGrid);
	FDM.SetDim({ FDMsizeX * scalarGridStepX, FDMsizeY * scalarGridStepY });
	auto dim = FDM.GetDim();
	cout << "Returned dim: " << dim << endl;
	FDM.bForceCPU = bForceCPU;

	Float RickerZero = RickerWavelet(0);

	FDM.AddSolutionInHistory([RickerZero](uint x, uint y)->Float
		{
			Float fx = Float(x) / FDMsizeX;
			Float fy = Float(y) / FDMsizeY;
			Float d2 = 0;

			if (fx < startOffset)
				d2 = (startOffset - fx) * (startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else if (fx > 1.0 - startOffset)
				d2 = (1.0 - startOffset - fx) * (1.0 - startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else
				d2 = (startOffset - fy) * (startOffset - fy) * FDMsizeY;

			return RickerWavelet(d2) / RickerZero;
		});
	FDM.AddSolutionInHistory([RickerZero, &velGrid](uint x, uint y)->Float
		{
			Float fx = Float(x) / FDMsizeX;
			Float fy = Float(y - FDMdeltaTime * velGrid.At(x, y) / scalarGridStepY) / FDMsizeY;
			Float d2 = 0;

			if (fx < startOffset)
				d2 = (startOffset - fx) * (startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else if (fx > 1.0 - startOffset)
				d2 = (1.0 - startOffset - fx) * (1.0 - startOffset - fx) * FDMsizeX + (startOffset - fy) * (startOffset - fy) * FDMsizeY;
			else
				d2 = (startOffset - fy) * (startOffset - fy) * FDMsizeY;

			return RickerWavelet(d2) / RickerZero;
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
	
	// ===================================
	// Decide Raycaster

	RaycastSystemType rs;

	rs.bInsertRays = bInsertRays;
	rs.bEvaluateDeltaTime = false;
	if (bAdaptiveTimestep) 
	{
		cout << "Adaptive timestep is disabled during hybrid mode." << endl;
	}
	rs.max_distance_parting = maxDistanceParting;
	rs.collapse_distance = collapseDistance;
	rs.bEnclosedShape = false;
	if (bCastAsSphere) 
	{
		cout << "Cast as sphere is not supported during hybrid mode. Falling back to plane cast." << endl;
	}
	rs.SetDeltaTime(FDMdeltaTime);
	if (FDMdeltaTime != ::deltaTime) 
	{
		cout << "Using FDM delta time: " << FDMdeltaTime << endl;
	}
	rs.maxSubstepping = maxSubstepping;
	rs.smoothingSteps = smoothingSteps;
	rs.amplitude_penalty = amplitudePenalty;
	rs.bFastDifferential = bFastDiff;

	if (bLoadGrid) 
		if (!rs.LoadScalarGrid(velGrid, Vector2{ scalarGridStepX, scalarGridStepY }))
		cout << "Failed to load scalar grid" << endl;

	if (gridInterpolation == "Linear")
		rs.samplerType = ScalarGridSamplerType::Linear;
	if (gridInterpolation == "Cubic")
		rs.samplerType = ScalarGridSamplerType::Cubic;
	if (gridInterpolation == "Hermite")
		rs.samplerType = ScalarGridSamplerType::Hermite;
	if (gridInterpolation == "BSpline")
		rs.samplerType = ScalarGridSamplerType::BSpline;

	rs.Create(rayNum, [](uint n, void* rs_in)->Ray2DT
		{
			RaycastSystemType* rs = (RaycastSystemType*)rs_in;
			Ray2DT ray;
			if (bLoadGrid)
			{
				Float alpha = Float(n) / rayNum * (1.0 - 2.0 * startOffset) + startOffset;
				ray.x(
					{ 
						FDMsizeX * scalarGridStepX * alpha,
						FDMsizeY * scalarGridStepY * startOffset
					});
				ray.p(Vector2({ 0, 1 }) / rs->C(ray.x()));
			}
			else
			{
				ray.x({ Float(n) / rayNum, startOffset });
				ray.p(Vector2({ 0, 1 }) / rs->C(ray.x()));
			}
			return ray;
		});

	// ======================================
	// Bind events

	FDM.Subscribe_OnSolution(
		// Decide
		[](int index)->bool { return index % writeSkip == 0; },
		// Write
		[](int index, decltype(FDM)::VectorX* sol)->void
		{
			lock_guard<mutex> FDMDataMutex(writerWorkerData.gridDataMutex);
			writerWorkerData.gridSnapshots.push(*sol);
		});
	rs.Subscribe_OnSolution([](int index, typename RaycastSystemType::Snapshot* snapshot)->void
		{
			if (index % writeSkip == 0) 
			{
				lock_guard<mutex> FDMDataMutex(writerWorkerData.rayDataMutex);
				typename RaycastSystemType::Snapshot* snapshotCopy = new RaycastSystemType::Snapshot();
				(*snapshotCopy) = (*snapshot);
				writerWorkerData.raySnapshots.push(snapshotCopy);
			}
		});

	// ======================================
	// Start writer thread
	writerWorkerData.bRunWriterLoop = true;
	thread writer;
	writer = thread(WriterWorker<RaycastSystemType>);

	rs.SimulateAsync(FDMsolTime);
	// Run "super mega safe" side thread to solve grids
	// Might fail unpredictably
	bool bFDMIsRunning = true;
	thread FDMSolverThread([&bFDMIsRunning]()->void { FDM.Solve(FDMsolTime); bFDMIsRunning = false; });

	// While solving is in progress
	auto messageSS = stringstream();
	messageSS.precision(2);
	
	while (rs.asyncStatus.bWorking || bFDMIsRunning) 
	{
		cout << '\r';

		messageSS.str(std::string());
		messageSS << "T: " << rs.asyncStatus.current_time << '\t'
			<< "Stp:" << rs.asyncStatus.steps_num << '\t'
			<< "Rays:" << rs.asyncStatus.rays_num << '\t' 
			<< "FDMT:" << FDM.currentSimTime << '\t'
			<< "Wrtq:" << max(writerWorkerData.raySnapshots.size(), writerWorkerData.gridSnapshots.size()) << "        ";

		cout << messageSS.str() << '\r';

		this_thread::sleep_for(250ms);
	}
	cout << endl;
	cout << "Finalizing. FDM running status: " << bFDMIsRunning << endl;
	writerWorkerData.bRunWriterLoop = false;
	if (writerWorkerData.gridSnapshots.size() == 0 && writerWorkerData.raySnapshots.size() == 0)
		cout << "Queue has been exhausted" << endl;
	else
		cout << "Queue had leftovers" << endl;
	writer.join();
	FDMSolverThread.join(); // Never finishes???

	FDM.Clear();

	cout << "Apparently, done." << endl;
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

	cout << "Floating point type size: " << sizeof(Float) << endl;

	Configuire();

	if (solverType == "FiniteDifference") 
	{
		if (bUseScalarGrid)
			RunFDM_Marm();
		else
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
	if (solverType == "Hybrid") 
	{
		if (integratorType == "Euler")
		{
			RunHybrid<RaycastWR<Float, Ray2D, IntegratorEuler>>();
		}
		if (integratorType == "RK4")
		{
			RunHybrid<RaycastWR<Float, Ray2D, IntegratorRK4Expl>>();
		}
		if (integratorType == "GaussLegendre")
		{
			RunHybrid<RaycastWR<Float, Ray2D, IntegratorGaussLegendre>>();
		}
	}
	if (solverType == "ScalarGridTest") 
	{
		ScalarGridTest();
		//BSplineTest();
	}

	cout << "Press enter to continue" << endl;
	cin.get();
	return 0;
}

