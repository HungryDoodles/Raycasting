#pragma once

#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <regex>

#include <Sparse>
#include <IterativeLinearSolvers>

#include "ScalarGrid.h"
#include <omp.h>
//#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl.hpp>

#undef max
#undef min

// 2D only yet
template<typename T>
class FiniteDifferenceSolver sealed
{
public:
	typedef Eigen::Matrix<uint, 2, 1> Vector2ui;
	typedef Eigen::Matrix<T, 2, 1> Vector2;
	typedef Eigen::SparseMatrix<T> Sparse;
	typedef Eigen::Matrix<T, -1, 1> VectorX;

	FiniteDifferenceSolver();
	FiniteDifferenceSolver(const Vector2ui& size);

	// Clear all data and reallocate
	void Resize(const Vector2ui& newSize);
	// Set dimension lengths to establish the physcial size of the system
	void SetDim(const Vector2& newDim);
	// Sets thenew size of the PML layer. CLEARS THE DATA
	void SetPMLSize(uint newSize);
	// Clear all data
	void Clear();
	// Self-explanatory
	void SetTimeStep(T newTimeStep);
	// Fills velocity grid using a formula
	void FillVelocityGrid(std::function<T(uint x, uint y)> c);
	// Loads velocity grid from a binary
	bool LoadVelocityGrid(const std::string& filename, uint sizeX, uint sizeY);
	// Loads velocity grid from other velocity grid
	bool LoadVelocityGrid(const ScalarGrid<T>& other);
	// Adds a layer of previous state that will account for the movement. Note: coordinates are indices and not physical coordinates
	void AddSolutionInHistory(std::function<T(uint x, uint y)> value);
	// Build sparse matrix
	//void Build(bool bWriteMatrix = false);
	// Build vector B
	//void BuildRHS(bool bWriteColumn = false);
	// Solve untile the timepoint T or until the maximum number of steps reached
	void Solve(T Time, long long max_steps = std::numeric_limits<long long>::max());

	// Initialize OpenCL. Returns true on success
	bool InitOpenCL();
	// Sets and inits buffers from solution history present on the host. Returns true on success
	bool SetBuffers();

	// Getters

	Vector2ui GetSize() const;
	Vector2 GetDim() const;
	size_t GetSolutionsCount() const;
	const VectorX* GetSolutionAt(size_t index) const;
	// Get last solution and remove it from the array
	bool PopLastSolution(VectorX& out);

	// Events handling

	// Subscribe for when the solver finishes step and gets the solution data
	// The function on the other side receives ordered index of the solution (from steps counter) and pointer to the vector.
	// Forces solver to purge needless part of the solution history to save some memory
	void Subscribe_OnSolution(std::function<bool(int)> predicator, std::function<void(int, VectorX*)> newNotification);
	void Unsubscribe_OnSolution();

	bool bForceCPU = true;
	T currentSimTime = 0;

private:
	
	// Teh mAtRiX
	Sparse m;
	// RHS
	VectorX b;

	std::vector<VectorX> solutions;

	T timestep = T(-1.0);
	uint PMLsize = 30;
	Vector2ui size = { 0, 0 };
	Vector2 dim = { 1.0, 1.0 };

	bool bVelGridLoaded = false;
	ScalarGrid<T> velGrid;

	bool bUseNotification = false;
	std::function<void(int, VectorX*)> notify_OnSolution;
	std::function<bool(int)> predicator_OnSolution;


	// CL stuff
	cl::Platform clPlatform;
	cl::Device clDevice;
	cl::Context clContext;
	cl::Program::Sources clSources;
	cl::Program clProgram;
	bool bClInitialized = false;

	int bufferLoopPosition = 0;
	cl::Buffer clBufferLoop[3];
	cl::Buffer clVelocityBuffer;
};





template<typename T>
inline FiniteDifferenceSolver<T>::FiniteDifferenceSolver()
{}

template<typename T>
inline FiniteDifferenceSolver<T>::FiniteDifferenceSolver(const Vector2ui& size) : size(size)
{
	velGrid.Resize(size(0), size(1));
}

template<typename T>
inline void FiniteDifferenceSolver<T>::Resize(const Vector2ui& newSize)
{
	size = newSize;
	velGrid.Resize(size(0), size(1));
	solutions.clear();
}

template<typename T>
inline void FiniteDifferenceSolver<T>::SetDim(const Vector2& newDim)
{
	if(newDim(0) != 0 && newDim(1) != 0)
		dim = newDim;
}

template<typename T>
inline void FiniteDifferenceSolver<T>::SetPMLSize(uint newSize)
{
	if (newSize > 2) 
	{
		Clear();
		PMLsize = newSize;
	}
}

template<typename T>
inline void FiniteDifferenceSolver<T>::Clear()
{
	velGrid.Clear();
	solutions.clear();
	Unsubscribe_OnSolution();

	clBufferLoop[0] = cl::Buffer();
	clBufferLoop[1] = cl::Buffer();
	clBufferLoop[2] = cl::Buffer();
	clVelocityBuffer = cl::Buffer();
	clProgram = cl::Program();
	clSources.clear();
	clContext = cl::Context();
	clDevice = cl::Device();
	clPlatform = cl::Platform();
	bClInitialized = false;
}

template<typename T>
inline void FiniteDifferenceSolver<T>::SetTimeStep(T newTimeStep)
{
	if (newTimeStep > 0)
		timestep = newTimeStep;
}

template<typename T>
inline void FiniteDifferenceSolver<T>::FillVelocityGrid(std::function<T(uint x, uint y)> c)
{
#pragma omp parallel for
	for (int y = 0; y < int(size(1)); ++y)
		for (int x = 0; x < int(size(0)); ++x)
		{
			velGrid.At(x, y) = c(x, y);
		}
	bVelGridLoaded = true;
}

template<typename T>
inline bool FiniteDifferenceSolver<T>::LoadVelocityGrid(const std::string& filename, uint sizeX, uint sizeY)
{
	if (sizeX == 0 || sizeY == 0)
		return false;
	Clear();
	size(0) = sizeX;
	size(1) = sizeY;
	bVelGridLoaded = velGrid.ReadFromBinary(filename, sizeX, sizeY);
	return bVelGridLoaded;
}

template<typename T>
inline bool FiniteDifferenceSolver<T>::LoadVelocityGrid(const ScalarGrid<T>& other)
{
	Clear();
	size(0) = other.GetSize(0);
	size(1) = other.GetSize(1);
	velGrid = other;
	bVelGridLoaded = true;
	return true;
}

template<typename T>
inline void FiniteDifferenceSolver<T>::AddSolutionInHistory(std::function<T(uint x, uint y)> value)
{
	if (size(0) == 0 || size(1) == 0)
		return;

	solutions.push_back(VectorX(size(0) * size(1)));
	auto& sol = solutions.back();
#pragma omp parallel for
	for (int y = 0; y < int(size(1)); ++y)
		for (int x = 0; x < int(size(0)); ++x)
		{
			sol[x + y * size(0)] = value(x, y);
		}
}


template<typename T>
inline void FiniteDifferenceSolver<T>::Solve(T max_time, long long max_steps)
{
	cout << "Solving until " << max_time << " or until " << max_steps << " of steps reached." << endl;
	cout << "Field grid size: (" << size(0) << ", " << size(1) << ")" << endl;
	cout << "Field physical size: (" << dim(0) << ", " << dim(1) << ")" << endl;

	cout << "Creating CL kernel functors" << endl;
	cl::CommandQueue queue(clContext, clDevice);
	cl_int err;
	//auto stepKernel = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&>(clProgram, "Step", &err);
	cl::Kernel stepKernel = cl::Kernel(clProgram, "Step", &err);
	if (err != CL_SUCCESS) 
	{
		cerr << "Kernel initialization has fucked up" << endl;
		return;
	}
	err = stepKernel.setArg(3, clVelocityBuffer);
	if (err != CL_SUCCESS)
	{
		cerr << "Velocity buffer has fucked up" << endl;
		return;
	}

	size_t maxGroupSize = 0;
	clDevice.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxGroupSize);
	cout << "Device max group size: " << maxGroupSize << endl;

	size_t dims[3];
	stepKernel.getWorkGroupInfo(clDevice, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, &dims);
	cout << "Work group size: " << dims[0] << endl;
	uint num_workers = (size[0] + PMLsize * 2) * (size[1] + PMLsize * 2);
	if (num_workers % dims[0] != 0)
		num_workers = (num_workers / dims[0] + 1) * dims[0];
	cout << "Workers num: " << num_workers << endl;
	cl::NDRange range = cl::NDRange(num_workers);

	uint numSteps = 0;
	T simTime = 0;
	int outputCounter = 0;

	while (simTime < max_time && numSteps++ < max_steps) 
	{
		currentSimTime = simTime;
		// Generate RHS
		cl_int ret = 0;
		ret = stepKernel.setArg(0, clBufferLoop[bufferLoopPosition]);
		ret = stepKernel.setArg(1, clBufferLoop[(bufferLoopPosition + 2) % 3]);
		ret = stepKernel.setArg(2, clBufferLoop[(bufferLoopPosition + 1) % 3]);
		if (ret != CL_SUCCESS)
			cerr << "Buffers error: " << ret << endl;

		ret = queue.enqueueNDRangeKernel(stepKernel, cl::NullRange, range, cl::NDRange(dims[0]));
		if (ret != CL_SUCCESS)
			cerr << "Kernel error: " << ret << endl;

		//ret = queue.finish();

		if (bUseNotification) if (predicator_OnSolution(numSteps))
		{
			ret = queue.finish();
			if (ret != CL_SUCCESS)
				cerr << "Queue error: " << ret << endl;
			cl_int readResult = queue.enqueueReadBuffer(clBufferLoop[bufferLoopPosition], CL_TRUE, 0, (size[0] + 2 * PMLsize) * (size[1] + 2 * PMLsize) * sizeof(T), b.data());
			if (readResult != CL_SUCCESS)
				cerr << "Buffer read failed" << endl;
			//VectorX centralSolution = b.block(PMLsize, PMLsize, size(0), size(1));
			notify_OnSolution(numSteps, &b);
		}

		bufferLoopPosition = (bufferLoopPosition + 1) % 3;
		simTime += timestep;
	}
}

template<typename T>
inline typename FiniteDifferenceSolver<T>::Vector2ui FiniteDifferenceSolver<T>::GetSize() const
{
	return size;
}

template<typename T>
inline typename FiniteDifferenceSolver<T>::Vector2 FiniteDifferenceSolver<T>::GetDim() const
{
	return dim;
}

template<typename T>
inline size_t FiniteDifferenceSolver<T>::GetSolutionsCount() const
{
	return solutions.size();
}

template<typename T>
inline const typename FiniteDifferenceSolver<T>::VectorX* FiniteDifferenceSolver<T>::GetSolutionAt(size_t index) const
{
	if (index >= solutions.size()) return nullptr;
	return &solutions[index];
}

template<typename T>
inline bool FiniteDifferenceSolver<T>::PopLastSolution(VectorX& out)
{
	if (solutions.size() < 0)
		return false;

	out = solutions.back();
	solutions.pop_back();
	return true;
}

template<typename T>
inline void FiniteDifferenceSolver<T>::Subscribe_OnSolution(std::function<bool(int)> predicator, std::function<void(int, VectorX*)> newNotification)
{
	bUseNotification = true;
	predicator_OnSolution = predicator;
	notify_OnSolution = newNotification;
	b.resize((size[0] + PMLsize * 2) * (size[1] + PMLsize * 2));
}

template<typename T>
inline void FiniteDifferenceSolver<T>::Unsubscribe_OnSolution()
{
	bUseNotification = false;
	notify_OnSolution = std::function<void(int, VectorX*)>();
	predicator_OnSolution = std::function<bool(int)>();
	b = VectorX();
}

// OPENCL STUFF

template<typename T>
inline bool FiniteDifferenceSolver<T>::InitOpenCL()
{
	// Check if can initialize at all
	if (!bVelGridLoaded || timestep < 0)
	{
		std::cerr << "Velocity grid and timestep must be set before initializing CL" << std::endl;
		return false;
	}

	// Init stuff: platforms
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty())
	{
		std::cerr << "OpenCL initialization failed: no available platform" << std::endl;
		return false;
	}
	clPlatform = platforms.front();

	// Init stuff: devices
	std::vector<cl::Device> devices;
	clPlatform.getDevices(bForceCPU ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU, &devices);

	if (devices.empty())
	{
		std::cerr << "No available GPUs were found, rolling back to CPU code" << std::endl;
		clPlatform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
		if (devices.empty())
		{
			std::cerr << "No available OpenCL devices were found" << std::endl;
			return false;
		}
	}
	clDevice = devices.front();

	cl::STRING_CLASS device_name = "Unknown";
	clDevice.getInfo(CL_DEVICE_NAME, &device_name);
	std::cout << "Using CL device: " << device_name << std::endl;
	cl_device_fp_config config;
	clDevice.getInfo(CL_DEVICE_DOUBLE_FP_CONFIG, &config);
	if (config == CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM)
		std::cout << "Double precision is SUPPORTED" << std::endl;
	else
		std::cout << "Double precision is UNSUPPORTED" << std::endl;
	cl_ulong maxBufferSize = 0;
	clDevice.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxBufferSize);
	std::cout << "Max buffer size: " << maxBufferSize << ". (Planned buffer requires at least " << (1461 * 760 * sizeof(T)) << "bytes" << std::endl;

	// Init stuff: context
	clContext = cl::Context(clDevice);

	// Init stuff: Source
	std::ifstream kernelFile("FiniteDifferenceKernels.cl");
	if (!kernelFile.is_open())
	{
		cerr << "Failed to open FiniteDifferenceKernels.cl" << endl;
		return false;
	}
	std::string src = std::string(std::istreambuf_iterator<char>(kernelFile), std::istreambuf_iterator<char>());

	// Some deep shit levels of optimization by hard setting some values in the source code
	std::string timestepStr = std::to_string(timestep);
	std::string sizeXStr = std::to_string(size[0]);
	std::string sizeYStr = std::to_string(size[1]);
	std::string dimXStr = std::to_string(dim[0]);
	std::string dimYStr = std::to_string(dim[1]);
	std::string PMLSizeStr = std::to_string(PMLsize);
	std::string type = (sizeof(T) == 8) ? "double" : "float";

	src = std::regex_replace(src, std::regex("<@T>"), timestepStr.c_str());
	src = std::regex_replace(src, std::regex("<@SizeX>"), sizeXStr.c_str());
	src = std::regex_replace(src, std::regex("<@SizeY>"), sizeYStr.c_str());
	src = std::regex_replace(src, std::regex("<@DimX>"), dimXStr.c_str());
	src = std::regex_replace(src, std::regex("<@DimY>"), dimYStr.c_str());
	src = std::regex_replace(src, std::regex("<@Type>"), type.c_str());
	src = std::regex_replace(src, std::regex("<@PMLSize>"), PMLSizeStr.c_str());

	ofstream("Formatted_code.cl") << src;

	clSources.push_back({src.c_str(), src.length()});
	clProgram = cl::Program(clContext, clSources);

	// Init stuff: build program
	if (clProgram.build({ clDevice }) != CL_SUCCESS) 
	{
		std::cout << "Error building: " << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(clDevice) << std::endl;
		return false;
	}

	bClInitialized = true;
	return true;
}

template<typename T>
inline bool FiniteDifferenceSolver<T>::SetBuffers()
{
	if (!bClInitialized)
	{
		std::cerr << "Unnable to set buffers: CL is not initialized" << std::endl;
		return false;
	}

	if (solutions.size() < 2)
	{
		std::cerr << "Unnable to set buffers: initial solutions were not specified" << std::endl;
		return false;
	}

	if (!bVelGridLoaded)
	{
		std::cerr << "Unnable to set buffers: velocity grid is not loaded although it should have been at this point" << std::endl;
		return false;
	}

	if (solutions.size() > 2)
		solutions.erase(solutions.begin(), solutions.end() - 2);

	Vector2ui PMLCompleteSize = size + Vector2ui{ 2 * PMLsize, 2 * PMLsize };
	uint PMLCompleteCount = PMLCompleteSize(0) * PMLCompleteSize(1);
	vector<VectorX> PMLCompleteLayers;
	PMLCompleteLayers.resize(2);
	for (int i = 0; i < 2; ++i)
	{
		PMLCompleteLayers[i].resize(PMLCompleteCount);
		PMLCompleteLayers[i].fill(0);
		for (int j = 0; j < size(1); ++j)
			PMLCompleteLayers[i].segment(PMLsize + (j + PMLsize) * (size(0) + 2 * PMLsize), size(0)) = solutions[i].segment(j * size(0), size(0));
		//PMLCompleteLayers[i].block(PMLsize, PMLsize, size(0), size(1)) = solutions[i].block(0, 0, size(0), size(1));
	}

	clVelocityBuffer = cl::Buffer(clContext, CL_READ_ONLY_CACHE | CL_MEM_COPY_HOST_PTR, sizeof(T) * velGrid.GetDataSize(), velGrid.Data());

	cl_int err1 = 0, err2 = 0, err3 = 0;
	clBufferLoop[2] = cl::Buffer(clContext, CL_READ_WRITE_CACHE, sizeof(T) * PMLCompleteLayers[1].size(), nullptr, &err3);
	clBufferLoop[1] = cl::Buffer(clContext, CL_READ_WRITE_CACHE | CL_MEM_COPY_HOST_PTR, sizeof(T) * PMLCompleteLayers[1].size(), PMLCompleteLayers[1].data(), &err2);
	clBufferLoop[0] = cl::Buffer(clContext, CL_READ_WRITE_CACHE | CL_MEM_COPY_HOST_PTR, sizeof(T) * PMLCompleteLayers[0].size(), PMLCompleteLayers[0].data(), &err1);

	size_t buf0size = 0;
	clBufferLoop[0].getInfo(CL_MEM_SIZE, &buf0size);
	std::cout << "Buf 0 size: " << buf0size << ". Submitted size: " << sizeof(T) * PMLCompleteLayers[0].size() << endl;

	std::cout << "Buf 0 index: " << (int)clBufferLoop[0]() << std::endl;
	std::cout << "Buf 1 index: " << (int)clBufferLoop[1]() << std::endl;
	std::cout << "Buf 2 index: " << (int)clBufferLoop[2]() << std::endl;
	std::cout << "Vel buf index: " << (int)clVelocityBuffer() << std::endl;

	if (err1 != CL_SUCCESS || err2 != CL_SUCCESS || err3 != CL_SUCCESS) 
	{
		std::cerr << "Buffer set failed with codes: " << err1 << ", " << err2 << ", " << err3 << endl;
		return false;
	}

	bufferLoopPosition = 2;

	return true;
}



// LEGACY

/*
template<typename T>
inline void FiniteDifferenceSolver<T>::Build(bool bWriteMatrix)
{
	if (size(0) == 0 || size(1) == 0)
		return;

	uint Nx = size(0);
	uint Ny = size(1);
	uint N = Nx * Ny;
	T dx = dim(0) / (Nx - 1);
	T dy = dim(1) / (Ny - 1);
	T dx2 = dx * dx;
	T dy2 = dy * dy;
	T dt = timestep;
	T dt2 = timestep * timestep;

	m = Sparse();
	m.resize(N, N);

	using Tri = Eigen::Triplet<T>;
	std::vector<Tri> tris; // (row, col, value)
	tris.reserve(N * 5);

	//=============================================================================================================================================
	// If inside
#pragma omp parallel for
	for (int y = 1; y < Ny - 1; ++y)
		for (int x = 1; x < Nx - 1; ++x)
		{
			T c = T(0); // Central
			T l = T(0); // Left term
			T r = T(0); // Right term
			T b = T(0); // Bottom term
			T t = T(0); // Top term
			// For full formulation see Algorithm.docx

			// Gathering values C
			T c_c = velGrid.At(x, y); T c_l = velGrid.At(x-1, y); T c_r = velGrid.At(x+1, y); T c_t = velGrid.At(x, y-1); T c_b = velGrid.At(x, y+1);
			// Evaluating values q
			auto q_sol = [](T a, T b)->T { return T(2.0) / (T(1.0) / (a * a) + T(1.0) / (b * b)); };
			T q_l = q_sol(c_c, c_l); T q_r = q_sol(c_c, c_r); T q_t = q_sol(c_c, c_t); T q_b = q_sol(c_c, c_b);

			// Summarize
			c = -(-q_r - q_l) / dx2 + -(-q_t - q_b) / dy2 + T(2) / dt2;
			l = -q_l / dx2;
			r = -q_r / dx2;
			t = -q_t / dy2;
			b = -q_b / dy2;

			int i = x + y * Nx;

			#pragma omp critical
			{
				//tris.push_back({ int(i), int(i + 0), T(1) });
				tris.push_back({ int(i), int(i + 0), c });
				tris.push_back({ int(i), int(i - 1), l });
				tris.push_back({ int(i), int(i + 1), r });
				tris.push_back({ int(i), int(i - Nx), t });
				tris.push_back({ int(i), int(i + Nx), b });
			}
		}

	bool bReflect = true;
#pragma omp parallel sections num_threads(4)
	{
	//=============================================================================================================================================
	// If left side
#pragma omp section
		{ // X = 0
			for (int y = 1; y < Ny - 1; ++y)
			{
				T c = T(0);
				T r = T(0);
				T rr = T(0);
				T t = T(0);
				T b = T(0);

				T c_c = velGrid.At(0, y);
				c = T(2) / (c_c * c_c * dt2) - T(3) / (T(4) * c_c * dt * dx) + T(1) / (dy2);
				r = T(3) / (c_c * dt * dx);
				rr = T(-1) / (T(4) * c_c * dt * dx);
				t = b = T(-1) / (T(2) * dy2);

				uint i = 0 + y * Nx;

				#pragma omp critical
				{
					if (bReflect)
						tris.push_back({ int(i), int(i + 0), T(1) });
					else
					{
						tris.push_back({ int(i), int(i + 0), c });
						tris.push_back({ int(i), int(i + 1), r });
						tris.push_back({ int(i), int(i + 2), rr });
						tris.push_back({ int(i), int(i - Nx), t });
						tris.push_back({ int(i), int(i + Ny), b });
					}
				}
			}
		}
	//=============================================================================================================================================
	// If right side
#pragma omp section
		{ // X = Lx
			for (int y = 1; y < Ny - 1; ++y)
			{
				T c = T(0);
				T l = T(0);
				T ll = T(0);
				T t = T(0);
				T b = T(0);

				T c_c = velGrid.At(Nx - 1, y);
				c = T(2) / (c_c * c_c * dt2) + T(3) / (T(4) * c_c * dt * dx) + T(1) / (dy2);
				l = T(-3) / (c_c * dt * dx);
				ll = T(1) / (T(4) * c_c * dt * dx);
				t = b = T(-1) / (T(2) * dy2);

				uint i = (Nx - 1) + y * Nx;

				#pragma omp critical
				{
					if (bReflect)
						tris.push_back({ int(i), int(i + 0), T(1) });
					else
					{
						tris.push_back({ int(i), int(i - 1), l });
						tris.push_back({ int(i), int(i - 2), ll });
						tris.push_back({ int(i), int(i - 0), c });
						tris.push_back({ int(i), int(i - Nx), t });
						tris.push_back({ int(i), int(i + Nx), b });
					}
				}
			}
		}
	//=============================================================================================================================================
	// If bottom
#pragma omp section
		{ // Y = Ly
			for (int x = 1; x < Nx - 1; ++x)
			{
				T c = T(0);
				T t = T(0);
				T tt = T(0);
				T l = T(0);
				T r = T(0);

				T c_c = velGrid.At(x, Ny - 1);
				c = T(2) / (c_c * c_c * dt2) + T(3) / (T(4) * c_c * dt * dx) + T(1) / (dy2);
				t = T(-3) / (c_c * dt * dy);
				tt = T(1) / (T(4) * c_c * dt * dy);
				l = r = T(-1) / (T(2) * dx2);

				uint i = x + (Ny - 1) * Nx;

				#pragma omp critical
				{
					if (bReflect)
						tris.push_back({ int(i), int(i + 0), T(1) });
					else
					{
						tris.push_back({ int(i), int(i + 0), c });
						tris.push_back({ int(i), int(i - Ny), t });
						tris.push_back({ int(i), int(i - 2 * Ny), tt });
						tris.push_back({ int(i), int(i - 1), l });
						tris.push_back({ int(i), int(i + 1), r });
					}
				}
			}
		}
	//=============================================================================================================================================
	// If top
#pragma omp section
		{ // Y = 0
			for (int x = 1; x < Nx - 1; ++x)
			{
				T c = T(0);
				T b = T(0);
				T bb = T(0);
				T l = T(0);
				T r = T(0);

				T c_c = velGrid.At(x, 0);
				c = T(2) / (c_c * c_c * dt2) - T(3) / (T(4) * c_c * dt * dx) + T(1) / (dy2);
				b = T(3) / (c_c * dt * dy);
				bb = T(-1) / (T(4) * c_c * dt * dy);
				l = r = T(-1) / (T(2) * dx2);

				uint i = x + (0) * Nx;

				#pragma omp critical
				{
					if (bReflect)
						tris.push_back({ int(i), int(i + 0), T(1) });
					else
					{
						tris.push_back({ int(i), int(i + 0), c });
						tris.push_back({ int(i), int(i + Ny), b });
						tris.push_back({ int(i), int(i + 2 * Ny), bb });
						tris.push_back({ int(i), int(i - 1), l });
						tris.push_back({ int(i), int(i + 1), r });
					}
				}
			}
		}
	}

	// Fixed values at the very corners
	tris.push_back({ 0, 0, T(1) });
	tris.push_back({ int(Nx - 1), int(Nx - 1), T(1) });
	tris.push_back({ int((Ny - 1) * Nx), int((Ny - 1) * Nx), T(1) });
	tris.push_back({ int(N - 1), int(N - 1), T(1) });

	m.setFromTriplets(tris.begin(), tris.end());
	m.makeCompressed();

	if (bWriteMatrix)
		std::ofstream("matrix.debug.txt") << m;
}
*/
/*
template<typename T>
inline void FiniteDifferenceSolver<T>::BuildRHS(bool bWriteColumn)
{
	if (size(0) == 0 || size(1) == 0)
		return;

	while (solutions.size() < 3)
		solutions.push_back(VectorX::Zero(size(0) * size(1)));

	b.resize(size(0) * size(1));

	auto& s0 = *(solutions.end() - 1);
	auto& s1 = *(solutions.end() - 2);
	auto& s2 = *(solutions.end() - 3);

	uint Nx = size(0);
	uint Ny = size(1);
	uint N = Nx * Ny;
	T dx = dim(0) / (Nx - 1);
	T dy = dim(1) / (Ny - 1);
	T dx2 = dx * dx;
	T dy2 = dy * dy;
	T dt = timestep;
	T dt2 = timestep * timestep;

#pragma omp parallel for
	for (int y = 1; y < Ny - 1; ++y)
		for (int x = 1; x < Nx - 1; ++x)
		{
			uint i = x + y * Nx;

			b(i) = -(T(-5) * s0[i] + T(4) * s1[i] - s2[i]) / dt2;
			//b(i) = (T(-2) * s0[i] + s1[i]) / dt2;
			/*
			T c, l, r, t, bot;
			// Gathering values C
			T c_c = velGrid.At(x, y); T c_l = velGrid.At(x - 1, y); T c_r = velGrid.At(x + 1, y); T c_t = velGrid.At(x, y - 1); T c_b = velGrid.At(x, y + 1);
			// Evaluating values q
			auto q_sol = [](T a, T b)->T { return T(2.0) / (T(1.0) / (a * a) + T(1.0) / (b * b)); };
			T q_l = q_sol(c_c, c_l); T q_r = q_sol(c_c, c_r); T q_t = q_sol(c_c, c_t); T q_b = q_sol(c_c, c_b);

			// Summarize
			c = (-q_r - q_l) / dx2 + (-q_t - q_b) / dy2 + T(2) / dt2;
			l = q_l / dx2;
			r = q_r / dx2;
			t = q_t / dy2;
			bot = q_b / dy2;
			b(i) = (l * s0[i - 1] + r * s0[i + 1] + t * s0[i - Nx] + bot * s0[i + Nx] + c * s0[i]) * dt2 - s1[i];
		}

	bool bRelfect = true;
#pragma omp parallel sections num_threads(4)
	{
		//=============================================================================================================================================
		// If left side
#pragma omp section
		{ // X = 0
			for (int y = 1; y < Ny - 1; ++y)
			{
				uint i = 0 + y * Nx;

				T c_c = velGrid.At(0, y);
				b(i) = -((T(-5) * s0[i] + T(4) * s1[i] - s2[i]) / (dt2 * c_c * c_c)
					+ (T(3) * s0[i] - T(4) * s1[i + 1] + s2[i + 2]) / (T(4) * dt * dx * c_c));
				if (bRelfect) b(i) = 0; // FORCE REFLECTION
			}
		}
		//=============================================================================================================================================
		// If right side
#pragma omp section
		{ // X = Lx
			for (int y = 1; y < Ny - 1; ++y)
			{
				uint i = (Nx - 1) + y * Nx;

				T c_c = velGrid.At(Nx - 1, y);
				b(i) = -((T(-5) * s0[i] + T(4) * s1[i] - s2[i]) / (dt2 * c_c * c_c)
					+ (T(-3) * s0[i] + T(4) * s1[i - 1] - s2[i - 2]) / (T(4) * dt * dx * c_c));
				if (bRelfect) b(i) = 0; // FORCE REFLECTION
			}
		}
		//=============================================================================================================================================
		// If bottom
#pragma omp section
		{ // Y = Ly
			for (int x = 1; x < Nx - 1; ++x)
			{
				uint i = x + (Ny - 1) * Nx;

				T c_c = velGrid.At(x, Ny - 1);
				b(i) = -((T(-5) * s0[i] + T(4) * s1[i] - s2[i]) / (dt2 * c_c * c_c)
					+ (T(-3) * s0[i] + T(4) * s1[i - Nx] - s2[i - 2 * Nx]) / (T(4) * dt * dy * c_c));
				if (bRelfect) b(i) = 0; // FORCE REFLECTION
			}
		}
		//=============================================================================================================================================
		// If top
#pragma omp section
		{ // Y = 0
			for (int x = 1; x < Nx - 1; ++x)
			{
				uint i = x + (0) * Nx;

				T c_c = velGrid.At(x, 0);
				b(i) = -((T(-5) * s0[i] + T(4) * s1[i] - s2[i]) / (dt2 * c_c * c_c)
					+ (T(3) * s0[i] - T(4) * s1[i + Nx] + s2[i + 2 * Nx]) / (T(4) * dt * dy * c_c));
				if (bRelfect) b(i) = 0; // FORCE REFLECTION
			}
		}
	}

	// Fixed values at the very corners
	b(0) = b(Nx - 1) = b((Ny - 1) * Nx) = b(N - 1) = 0;

	if (bWriteColumn)
		ofstream("RHS.debug.txt") << b;
}
*/