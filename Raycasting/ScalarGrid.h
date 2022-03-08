#pragma once
#include "Header.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <functional>

#include "lodepng/lodepng.h"
#include "gif/gif.h"
#include "src/interpolation.h"

// Assumes (0,1) interpolation between points p1 and p2
template<typename T>
inline T Lerp(T x, T p0, T p1) 
{
	return p0 * (1 - x) + p1 * x;
}
template<typename T>
inline T Cubic(T x, T p0, T p1, T p2, T p3)
{
	return p1 + 
		x * (T(-0.5) * p0 + T(0.5) * p2) + 
		x * x * (p0 - T(2.5) * p1 + T(2) * p2 - T(0.5) * p3) +
		x * x * x * (T(-0.5) * p0 + T(1.5) * p1 - T(1.5) * p2 + T(0.5) * p3);
}
template<typename T>
inline T IDW(T x_in, T p0, T p1, T p2, T p3) 
{
	T v[3] = { p1 - p0, p2 - p1, p3 - p2 };
	T t[4] = { 0 };
	t[1] = pow(1 + v[0] * v[0], T(0.25));
	t[2] = pow(1 + v[1] * v[1], T(0.25)) + t[1];
	t[3] = pow(1 + v[2] * v[2], T(0.25)) + t[2];
	T x = (T(1) - x_in) * t[1] + x_in * t[2];

	T A1, A2, A3, B1, B2, C;

	A1 = ( (t[1] - x) * p0 + (x - t[0]) * p1 ) / (t[1] - t[0]);
	A2 = ( (t[2] - x) * p1 + (x - t[1]) * p2 ) / (t[2] - t[1]);
	A3 = ( (t[3] - x) * p2 + (x - t[2]) * p3 ) / (t[3] - t[2]);

	B1 = ( (t[2] - x) * A1 + (x - t[0]) * A2 ) / (t[2] - t[0]);
	B2 = ( (t[3] - x) * A2 + (x - t[1]) * A3 ) / (t[3] - t[1]);

	C = ( (t[2] - x) * B1 + (x - t[1]) * B2 ) / (t[2] - t[1]);

	return C;
}
// Hermite implementation of the same CatmullRom
template<typename T>
inline T CatmullRomHermite(T x, T p0, T p1, T p2, T p3)
{
	T x3 = x * x * x;
	T x2 = x * x;
	return (T(2) * x3 - T(3) * x2 + T(1)) * p1 + 
		(x3 - T(2) * x2 + x) * (T(0.5) * (p2 - p0)) + 
		(T(-2) * x3 + 3 * x2) * p2 + 
		(x3 - x2) * (T(0.5) * (p3 - p1));
}
// Minimal slope Hermite interpolation
template<typename T>
inline T Hermite(T x, T p0, T p1, T p2, T p3)
{
	T x3 = x * x * x;
	T x2 = x * x;
	T leftSlope = (std::abs(p1 - p0) <= std::abs(p2 - p1)) ? (p1 - p0) : (p2 - p1);
	T rightSlope = (std::abs(p2 - p1) <= std::abs(p3 - p2)) ? (p2 - p1) : (p3 - p2);
	return (T(2) * x3 - T(3) * x2 + T(1)) * p1 +
		(x3 - T(2) * x2 + x) * leftSlope +
		(T(-2) * x3 + 3 * x2) * p2 +
		(x3 - x2) * rightSlope;
}
template<typename T>
inline T Hermite_dx(T x, T p0, T p1, T p2, T p3)
{
	T x3 = x * x * x;
	T x2 = x * x;
	T leftSlope = (std::abs(p1 - p0) <= std::abs(p2 - p1)) ? (p1 - p0) : (p2 - p1);
	T rightSlope = (std::abs(p2 - p1) <= std::abs(p3 - p2)) ? (p2 - p1) : (p3 - p2);
	return (T(6) * x2 - T(6) * x) * p1 +
		(T(3) * x2 - T(4) * x + T(1)) * leftSlope +
		(T(-6) * x2 + T(6) * x) * p2 +
		(T(3) * x2 - T(2) * x) * rightSlope;
}
// Naive implementation
template<typename T>
inline T BSplineBasisRecursive(T x, uint i, uint order, const std::vector<T>& t) 
{
	if (order == 0) 
	{
		if (x >= t[i] && x < t[i])
			return 1; // Why?
		return 0;
	}
	else 
	{
		T alpha1 = (x - t[i]) / (t[i + order] - t[i]);
		T alpha2 = (t[i + order + 1] - x) / (t[i + p + 1] - t[i + 1]);
		return alpha1 * BSplineBasisRecursive(x, i, order - 1, t, b) + alpha2 * BSplineBasisRecursive(x, i + 1, order - 1, t, b);
	}
}
// De Boor algorithm, but it only takes evenly spaced control points, space between points equals 1.
//  x -- time between points in the middle,
// c.size() -- number of points, should be at least p + 1
// p -- order
template<typename T>
inline T BSplineDeBoor(T x, uint p, const std::vector<T>& c) 
{
	static thread_local std::vector<T> d; // Keep in memory
	d.assign(c.begin(), c.end());

	for (int r = 1; r <= p; ++r) 
	{
		for (int j = p; j >= r; --j)
		{
			T alpha = (x - (j - int(p))) / ((j + 1 - r) - (j - int(p)));
			d[j] = (1 - alpha) * d[j - 1] + alpha * d[j];
		}
	}

	return d[p];
}


template<typename T>
inline T MinAbs(T a, T b) 
{
	return (std::abs(a) < std::abs(b)) ? a : b;
}

enum class ScalarGridSamplerType 
{
	Linear, Cubic, Hermite, BSpline
};

// A simple class that contains scalar data, provides interpolations and fancy utils
template<typename T>
class ScalarGrid
{
public:
	ScalarGrid();
	ScalarGrid(uint sizeX, uint sizeY);
	ScalarGrid(uint sizeX, uint sizeY, uint sizeZ);
	virtual ~ScalarGrid();

	// Warning: clears the data. Size of Z dimension becomes 1.
	void Resize(uint sizeX, uint sizeY);
	// Warning: clears the data
	void Resize(uint sizeX, uint sizeY, uint sizeZ);

	void Clear();
	// Reinit
	void Zero();

	uint GetSize(int dim) const;
	size_t GetDataSize() const { return data.size(); }
	T* GetData() const { return data.data(); }

	// Fills from binary file that contains no header information indicating size
	template<typename elemT = T>
	bool ReadFromBinary(const std::string& filename, uint sizeX, uint sizeY, uint sizeZ = 1, bool bFlipY = true);

	uint Dim(uint index) const;
	T At(int x, int y, int z = 0) const;
	T& At(int x, int y, int z = 0);
	T* Data();

	// Use sampler on underlying data to get values between elements
	virtual T Sample(ScalarGridSamplerType sampler, T x, T y, T z = 0) const;
	// Sample that takes derivative approximation. Quadruples computation cost.
	// Argument @index represents basis vector index (0 being X, 1 being Y etc.)
	virtual T SampleDerivative(ScalarGridSamplerType sampler, uint index, T x, T y, T z = 0) const;

	// Use the specified function on the whole field using the point funtion
	void Process(std::function<T(T, int, int, int)> func);

	// Performs 5x5 exponential kernel smoothing
	void ExpKernel();

	// Evaluate central difference of the current scalar grid in the specified dimension
	// 0 = X, 1 = Y, 2 = Z
	void CentralDifference(int dim);
	// Evaluate minimal difference: a numerical derivated chosen by the minimal absolute value between left- and right-sided difference formula
	void MinimalDifference(int dim);

	// ====== Fancy stuff

	// Writes into an ASCII stream using table format
	void WriteIntoStream(std::ofstream& out);

	// Projects a single point
	void Project_Point(T x, T y, T value);
	// Projects interpolated point
	void Project_Bilinear(T x, T y, T value);

	// Resample. x0, y0 - coordinates inside THIS grid. Scale is the coordinate multiplier for the OTHER grid
	void Resample(const ScalarGrid<T>& other, uint x0, uint y0, T scaleX, T scaleY, ScalarGridSamplerType sampler = ScalarGridSamplerType::Cubic);
	// Literally the only flip ever needed
	void FlipY();

	// Prepares to write into a gif
	void BeginGif(const std::string& filename, int delay = 16);
	// Closes gif
	void EndGif();
	// Writes current state into the gif
	// Interpolates values given in range [low] and [high] as corresponding rgb triplets 
	void WriteGifFrame(T low = 0, T high = 1, 
		uint8_t r0 = 0, uint8_t g0 = 0, uint8_t b0 = 0, 
		uint8_t r1 = 255, uint8_t g1 = 255, uint8_t b1 = 255);
	// Writes current state into the gif
	// Uses arg @law to decide color for given value
	// Color format is presented in rgba format. Alpha does nothing.
	void WriteGifFrame(std::function<uint32_t(float)> law);

	// Writes a gif frame into target
	// Uses n + 1 laws to decide color where law 0 is used to write target, alpha is ignored.
	// Uses n other laws to write layers in their respective order by mixing recursively according to alpha.
	void WriteGifFrameLayered(std::function<uint32_t(float)>* laws, const ScalarGrid<T>* layers, int n);

protected:
	void FillSquare(int X, int Y, int size) const;
	void FillCube(int X, int Y, int Z, int size) const;
	mutable T c[4][4][4]; // A complete cube of vaules for interpolation purposes
	mutable T s[4][4]; // A square of values
	inline const static T kern[5][5] =
	{ 
		{ 9.11382427473493E-09,		7.7837414819609137E-06,		7.3850082999351E-05,	7.7837414819609137E-06,		9.11382427473493E-09		}, 
		{ 7.7837414819609137E-06,	0.0066477726179070109,		0.063072310498879278,	0.0066477726179070109,		7.7837414819609137E-06		}, 
		{ 7.3850082999351E-05,		0.063072310498879278,		0.59841342060214908,	0.063072310498879278,		7.3850082999351E-05			}, 
		{ 7.7837414819609137E-06,	0.0066477726179070109,		0.063072310498879278,	0.0066477726179070109,		7.7837414819609137E-06		}, 
		{ 9.11382427473493E-09,		7.7837414819609137E-06,		7.3850082999351E-05,	7.7837414819609137E-06,		9.11382427473493E-09		} 
	};

	T LinearConv(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z = 0) const;
	T CubicConv(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z = 0) const;
	T BSpline(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z = 0) const;
	T HermiteConv(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z = 0) const;

private:
	uint sizeX;
	uint sizeY;
	uint sizeZ = 1;

	std::vector<T> data;

	// Gif stuff
	bool bWritingGif = false;
	GifWriter gifWriter;
	std::vector<uint8_t> gif_frame;
	int gif_delay;
};



template<typename T>
inline ScalarGrid<T>::ScalarGrid() : sizeX(0), sizeY(0), sizeZ(0)
{}

template<typename T>
inline ScalarGrid<T>::ScalarGrid(uint sizeX, uint sizeY) : sizeX(sizeX), sizeY(sizeY), sizeZ(1)
{
	if (sizeX == 0 || sizeY == 0 || sizeZ == 0)
	{
		sizeX = sizeY = sizeZ = 0;
		return;
	}
	uint size = sizeX * sizeY * sizeZ;
	data.resize(size, T());
}

template<typename T>
inline ScalarGrid<T>::ScalarGrid(uint sizeX, uint sizeY, uint sizeZ) : sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ)
{
	if (sizeX == 0 || sizeY == 0 || sizeZ == 0)
	{
		sizeX = sizeY = sizeZ = 0;
		return;
	}
	size = sizeX * sizeY * sizeZ;
	data.resize(size, T());
}

template<typename T>
inline ScalarGrid<T>::~ScalarGrid()
{
	if (bWritingGif)
		bWritingGif = false;
}

template<typename T>
inline void ScalarGrid<T>::Resize(uint sizeX, uint sizeY)
{
	EndGif();
	ScalarGrid<T>::sizeX = sizeX;
	ScalarGrid<T>::sizeY = sizeY;
	sizeZ = 1;

	size_t size = sizeX * sizeY * sizeZ;
	data.resize(size, T());
}

template<typename T>
inline void ScalarGrid<T>::Resize(uint sizeX, uint sizeY, uint sizeZ)
{
	EndGif();
	ScalarGrid<T>::sizeX = sizeX;
	ScalarGrid<T>::sizeY = sizeY;
	ScalarGrid<T>::sizeZ = sizeZ;

	uint size = sizeX * sizeY * sizeZ;
	data.resize(size, T());
}

template<typename T>
inline void ScalarGrid<T>::Clear()
{
	EndGif();
	data.clear();
	sizeX = 0;
	sizeY = 0;
	sizeZ = 0;
}

template<typename T>
inline void ScalarGrid<T>::Zero()
{
	for (auto& value : data)
		value = T(0);
}

template<typename T>
inline uint ScalarGrid<T>::GetSize(int dim) const
{
	switch (dim) 
	{
	case 0:
		return sizeX;
	case 1:
		return sizeY;
	case 2:
		return sizeZ;
	}
	return uint();
}

template<typename T>
inline uint ScalarGrid<T>::Dim(uint index) const
{
	switch (index) 
	{
	case 0:
		return sizeX;
	case 1:
		return sizeY;
	case 2:
		return sizeZ;
	}
	return 0;
}

template<typename T>
inline T ScalarGrid<T>::At(int x, int y, int z) const
{
	// Clamp
	if (x >= int(sizeX)) x = int(sizeX) - 1;
	if (y >= int(sizeY)) y = int(sizeY) - 1;
	if (z >= int(sizeZ)) z = int(sizeZ) - 1;
	if (x < 0) x = 0;
	if (y < 0) y = 0;
	if (z < 0) z = 0;

	return data[x + y * sizeX + z * sizeX * sizeY];
}

template<typename T>
inline T& ScalarGrid<T>::At(int x, int y, int z)
{
	// Clamp
	if (x >= int(sizeX)) x = int(sizeX) - 1;
	if (y >= int(sizeY)) y = int(sizeY) - 1;
	if (z >= int(sizeZ)) z = int(sizeZ) - 1;
	if (x < 0) x = 0;
	if (y < 0) y = 0;
	if (z < 0) z = 0;

	return data[x + y * sizeX + z * sizeX * sizeY];
}

template<typename T>
inline T* ScalarGrid<T>::Data()
{
	return data.data();
}

template<typename T>
T ScalarGrid<T>::Sample(ScalarGridSamplerType sampler, T x, T y, T z) const
{
	int fX = int(std::floor(x));
	int fY = int(std::floor(y));
	int fZ = int(std::floor(z));
	T alphaX = x - fX;
	T alphaY = y - fY;
	T alphaZ = z - fZ;

	T value = 0;

	switch (sampler) 
	{
	case ScalarGridSamplerType::Linear:
		if (sizeZ == 1) // Sample square
		{
			FillSquare(fX, fY, 2);

			value = (1.0 - alphaY) * ((1.0 - alphaX) * s[0][0] + alphaX * s[1][0]) +
				alphaY * ((1.0 - alphaX) * s[0][1] + alphaX * s[1][1]);
		}
		else // Sample cube
		{
			FillCube(fX, fY, fZ, 2);

			value = 
				((1.0 - alphaY) * ((1.0 - alphaX) * c[0][0][0] + alphaX * c[1][0][0]) +
				alphaY * ((1.0 - alphaX) * c[0][1][0] + alphaX * c[1][1][0])) * alphaZ
				+
				((1.0 - alphaY) * ((1.0 - alphaX) * c[0][0][1] + alphaX * c[1][0][1]) +
				alphaY * ((1.0 - alphaX) * c[0][1][1] + alphaX * c[1][1][1])) * alphaZ;
		}
		break;
	case ScalarGridSamplerType::Cubic:
		if (sizeZ == 1) // Sample square 4x4
			value = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY);
		else // Sample cube
			value = CubicConv(2, alphaX, alphaY, alphaZ, fX, fY, fZ);
		break;
	case ScalarGridSamplerType::BSpline:
		if (sizeZ == 1)
			value = BSpline(1, alphaX, alphaY, alphaZ, fX, fY);
		else
			value = BSpline(2, alphaX, alphaY, alphaZ, fX, fY, fZ);
		break;
	case ScalarGridSamplerType::Hermite:
		if (sizeZ == 1)
			value = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY);
		else
			value = HermiteConv(2, alphaX, alphaY, alphaZ, fX, fY, fZ);
	}

	return value;
}

template<typename T>
void Separate(T x, T y, T z, int& fX, int& fY, int& fZ, T& alphaX, T& alphaY, T& alphaZ) 
{
	fX = int(std::floor(x));
	fY = int(std::floor(y));
	fZ = int(std::floor(z));
	alphaX = x - fX;
	alphaY = y - fY;
	alphaZ = z - fZ;
}
template<typename T>
T ScalarGrid<T>::SampleDerivative(ScalarGridSamplerType sampler, uint index, T x, T y, T z) const
{
	// Fix it
	T dX = index == 0 ? d_h : 0;
	T dY = index == 1 ? d_h : 0;
	T dZ = index == 2 ? d_h : 0;

	int fX, fY, fZ;
	T alphaX, alphaY, alphaZ;

	T v0 = 0, v1 = 0, v2 = 0, v3 = 0;
	T value = 0;

	switch (sampler)
	{
	case ScalarGridSamplerType::Linear:
		if (sizeZ == 1) // Sample square
		{
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY);
		}
		else // Sample cube
		{
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = LinearConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);
		}
		break;
	case ScalarGridSamplerType::Cubic:
		if (sizeZ == 1)
		{ // Sample square 4x4
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY);
		}
		else
		{ // Sample cube
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = CubicConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);
		}
		break;
	case ScalarGridSamplerType::BSpline:
		if (sizeZ == 1)
		{
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY);
		}
		else
		{
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = BSpline(1, alphaX, alphaY, alphaZ, fX, fY, fZ);
		}
		break;
	case ScalarGridSamplerType::Hermite:
		if (sizeZ == 1)
		{
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY);
		}
		else
		{
			Separate(x - 2 * dX, y - 2 * dY, z - 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v0 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x - 1 * dX, y - 1 * dY, z - 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v1 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 1 * dX, y + 1 * dY, z + 1 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v2 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);

			Separate(x + 2 * dX, y + 2 * dY, z + 2 * dZ, fX, fY, fZ, alphaX, alphaY, alphaZ);
			v3 = HermiteConv(1, alphaX, alphaY, alphaZ, fX, fY, fZ);
		}
	}

	//value = (-v3 + 8*v2 - 8*v1 + v0) / (12 * d_h);
	
	value = (v2 - v1) / (2 * d_h);

	return value;
}

template<typename T>
inline void ScalarGrid<T>::Process(std::function<T(T, int, int, int)> func)
{
	for (int z = 0; z < sizeZ; ++z)
		for (int y = 0; y < sizeY; ++y)
			for (int x = 0; x < sizeX; ++x)
				data[x + y * sizeX + z * sizeX * sizeY] =
				func(data[x + y * sizeX + z * sizeX * sizeY], x, y, z);
}

template<typename T>
inline void ScalarGrid<T>::ExpKernel()
{
	std::vector<T> newData(data.size());

	// Failsafe
	T kernSum = 0;
	for (int kx = 0; kx < 5; ++kx)
		for (int ky = 0; ky < 5; ++ky)
			kernSum += kern[kx][ky];

	for (int y = 0; y < sizeY; ++y)
		for (int x = 0; x < sizeX; ++x)
		{
			T sum = 0;
			for (int kx = 0; kx < 5; ++kx)
				for (int ky = 0; ky < 5; ++ky)
					sum += kern[kx][ky] * At(x - 2 + kx, y - 2 + ky);
			newData[x + y * sizeX] = sum / kernSum;
		}

	data = newData;
}

template<typename T>
inline void ScalarGrid<T>::CentralDifference(int dim)
{
	std::vector<T> newData(data.size());

	for (int z = 0; z < sizeZ; ++z)
		for (int y = 0; y < sizeY; ++y)
			for (int x = 0; x < sizeX; ++x)
			{
				T diff = 0;
				T x0 = At(x + 1, y, z), x1 = At(x - 1, y, z);
				switch (dim) 
				{
				default:
				case 0: diff = (At(x + 1, y, z) - At(x - 1, y, z)) * T(0.5); break;
				case 1: diff = (At(x, y + 1, z) - At(x, y - 1, z)) * T(0.5); break;
				case 2: diff = (At(x, y, z + 1) - At(x, y, z - 1)) * T(0.5); break;
				}

				newData[x + y * sizeX + z * sizeX * sizeY] = diff;
			}

	data = newData;
}

template<typename T>
inline void ScalarGrid<T>::MinimalDifference(int dim)
{
	std::vector<T> newData(data.size());

	for (int z = 0; z < sizeZ; ++z)
		for (int y = 0; y < sizeY; ++y)
			for (int x = 0; x < sizeX; ++x)
			{
				T diff = 0;
				T x0 = At(x + 1, y, z), x1 = At(x - 1, y, z);
				switch (dim)
				{
				default:
				case 0: 
					diff = MinAbs(
						At(x, y, z) - At(x - 1, y, z), 
						At(x + 1, y, z) - At(x, y, z)); 
					break;
				case 1: 
					diff = MinAbs(
						At(x, y, z) - At(x, y - 1, z), 
						At(x, y + 1, z) - At(x, y, z)); 
					break;
				case 2: 
					diff = MinAbs(
						At(x, y, z) - At(x, y, z - 1), 
						At(x, y, z + 1) - At(x, y, z)); 
					break;
				}

				newData[x + y * sizeX + z * sizeX * sizeY] = diff;
			}

	data = newData;
}

template<typename T>
inline void ScalarGrid<T>::WriteIntoStream(std::ofstream& out)
{
	if (sizeZ == 1) 
	{
		for (int iy = 0; iy < sizeY; ++iy)
		{
			for (int ix = 0; ix < sizeX; ++ix)
				out << At(ix, iy, 0) << '\t';
			out << std::endl;
		}
	}
	else 
	{
		cout << "# Grid is in 3D format, write function is not supported" << std::endl;
	}
}

template<typename T>
inline void ScalarGrid<T>::Project_Point(T x, T y, T value)
{
	At(x, y, 0) += value;
}

template<typename T>
inline void ScalarGrid<T>::Project_Bilinear(T x, T y, T value)
{
	int X = std::floor(x);
	int Y = std::floor(y);
	T alphaX = x - X;
	T alphaY = y - Y;

	At(X + 0, Y + 0, 0) += (T(1) - alphaX) * (T(1) - alphaY) * value;
	At(X + 1, Y + 0, 0) += (alphaX) * (T(1) - alphaY) * value;
	At(X + 0, Y + 1, 0) += (T(1) - alphaX) * (alphaY) * value;
	At(X + 1, Y + 1, 0) += (alphaX) * (alphaY) * value;
}

template<typename T>
void ScalarGrid<T>::Resample(const ScalarGrid<T>& other, uint x0, uint y0, T scaleX, T scaleY, ScalarGridSamplerType sampler) 
{
	uint dimX = scaleX * (other.sizeX + 0.5);
	uint dimY = scaleY * (other.sizeY + 0.5);


#pragma omp parallel for
	for (int y = 0; y < min((uint)dimY, (uint)sizeY); ++y) 
	{
		for (int x = 0; x < min((uint)dimX, (uint)sizeX); ++x)
			data[x + y * sizeX] = other.Sample(sampler, (x - x0) / scaleX, (y - y0) / scaleY);
	}
}

template<typename T>
inline void ScalarGrid<T>::BeginGif(const std::string& filename, int delay)
{
	gif_frame.resize(sizeX * sizeY * 4);
	gif_delay = delay;
	GifBegin(&gifWriter, filename.c_str(), sizeX, sizeY, delay);
	bWritingGif = true;
}

template<typename T>
inline void ScalarGrid<T>::EndGif()
{
	if (bWritingGif)
	{
		GifEnd(&gifWriter);
		bWritingGif = false;
	}
}

template<typename T>
inline void ScalarGrid<T>::WriteGifFrame(T low, T high, uint8_t r0, uint8_t g0, uint8_t b0, uint8_t r1, uint8_t g1, uint8_t b1)
{
	if (!bWritingGif)
		return;

#pragma omp parallel for
	for (int iy = 0; iy < sizeY; ++iy)
		for (int ix = 0; ix < sizeX; ++ix) 
		{
			T alpha = (At(ix, iy, 0) - low) / (high - low);
			alpha = std::max(T(0), std::min(T(1), alpha));

			gif_frame[(ix + iy * sizeX) * 4 + 0] = uint8_t(alpha * r1 + (1 - alpha) * r0);
			gif_frame[(ix + iy * sizeX) * 4 + 1] = uint8_t(alpha * g1 + (1 - alpha) * g0);
			gif_frame[(ix + iy * sizeX) * 4 + 2] = uint8_t(alpha * b1 + (1 - alpha) * b0);
		}

	GifWriteFrame(&gifWriter, gif_frame.data(), sizeX, sizeY, gif_delay, 8, false);
}

template<typename T>
void ScalarGrid<T>::WriteGifFrame(std::function<uint32_t(float)> law) 
{
	if (!bWritingGif)
		return;

#pragma omp parallel for
	for (int iy = 0; iy < sizeY; ++iy)
		for (int ix = 0; ix < sizeX; ++ix)
		{
			uint32_t c = law(At(ix, iy));

			gif_frame[(ix + iy * sizeX) * 4 + 0] = uint8_t((c >>  0) & 0xff);
			gif_frame[(ix + iy * sizeX) * 4 + 1] = uint8_t((c >>  8) & 0xff);
			gif_frame[(ix + iy * sizeX) * 4 + 2] = uint8_t((c >> 16) & 0xff);
		}

	GifWriteFrame(&gifWriter, gif_frame.data(), sizeX, sizeY, gif_delay, 8, false);
}

inline uint8_t mix_alpha(uint8_t base, uint8_t paint, uint8_t alpha)
{
	uint16_t base16, paint16;
	base16 = ((uint16_t)base) * (255 - alpha);
	paint16 = ((uint16_t)paint) * alpha;
	return (base16 + paint16) / 255;
}
// Base alpha is ignored.
// Output alpha is always 255
inline uint32_t mix_alpha(uint32_t base, uint32_t paint) 
{
	uint8_t alpha = paint >> 24;
	uint32_t ret = 0;
	ret |= (uint32_t)mix_alpha(uint8_t(base >> 0), uint8_t(paint >> 0), alpha) << 0;
	ret |= (uint32_t)mix_alpha(uint8_t(base >> 8), uint8_t(paint >> 8), alpha) << 8;
	ret |= (uint32_t)mix_alpha(uint8_t(base >> 16), uint8_t(paint >> 16), alpha) << 16;
	ret |= 0xff000000;
	return ret;
}

template<typename T>
void ScalarGrid<T>::WriteGifFrameLayered(std::function<uint32_t(float)>* laws, const ScalarGrid<T>* layers, int n)
{
	if (!bWritingGif)
		return;

	if (n < 1)
		return;

#pragma omp parallel for
	for (int iy = 0; iy < sizeY; ++iy)
		for (int ix = 0; ix < sizeX; ++ix)
		{
			uint32_t c = laws[0](At(ix, iy));
			for (int i = 1; i < n + 1; ++i)
				c = mix_alpha(c, laws[i](layers[i - 1].At(ix, iy)));

			gif_frame[(ix + iy * sizeX) * 4 + 0] = uint8_t((c >> 0) & 0xff);
			gif_frame[(ix + iy * sizeX) * 4 + 1] = uint8_t((c >> 8) & 0xff);
			gif_frame[(ix + iy * sizeX) * 4 + 2] = uint8_t((c >> 16) & 0xff);
		}

	GifWriteFrame(&gifWriter, gif_frame.data(), sizeX, sizeY, gif_delay, 8, false);
}

template<typename T>
inline void ScalarGrid<T>::FillSquare(int X, int Y, int size) const
{
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			s[j][i] = At(X + j, Y + i);
}

template<typename T>
inline void ScalarGrid<T>::FillCube(int X, int Y, int Z, int size) const
{
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < size; ++j)
			for (int k = 0; k < size; ++k)
				c[k][j][i] = At(X + k, Y + j, Z + i);
}

template<typename T>
inline T ScalarGrid<T>::LinearConv(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z) const
{
	if (level > 2)
		level = 2;
	if (level <= 0)
	{
		return Lerp(alphaX,
			At(x + 0, y, z),
			At(x + 1, y, z));
	}
	else switch (level)
	{
	case 1:
		return Lerp(alphaY,
			LinearConv(level - 1, alphaX, alphaY, alphaZ, x, y + 0, z),
			LinearConv(level - 1, alphaX, alphaY, alphaZ, x, y + 1, z));
		break;
	case 2:
		return Lerp(alphaZ,
			LinearConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 0),
			LinearConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 1));
		break;
	}
	throw "Impossible branch reached";
}

template<typename T>
inline T ScalarGrid<T>::CubicConv(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z) const
{
	if (level > 2)
		level = 2;
	if (level <= 0) 
	{
		return Cubic(alphaX,
			At(x - 1, y, z),
			At(x + 0, y, z),
			At(x + 1, y, z),
			At(x + 2, y, z));
	}
	else switch (level)
	{
	case 1:
		return Cubic(alphaY,
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y - 1, z),
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y + 0, z),
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y + 1, z),
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y + 2, z));
		break;
	case 2:
		return Cubic(alphaZ,
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y, z - 1),
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 0),
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 1),
			CubicConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 2));
		break;
	}
	throw "Impossible branch reached";
}

template<typename T>
inline T ScalarGrid<T>::BSpline(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z) const
{
	if (level > 2)
		level = 2;
	if (level <= 0)
	{
		return ::BSplineDeBoor(alphaX, 5, std::vector<T>{
				At(x - 2, y, z),
				At(x - 1, y, z),
				At(x + 0, y, z),
				At(x + 1, y, z),
				At(x + 2, y, z),
				At(x + 3, y, z)});
	}
	else switch (level)
	{
	case 1:
		return ::BSplineDeBoor(alphaY, 5, std::vector<T>
		{
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y - 2, z),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y - 1, z),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y + 0, z),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y + 1, z),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y + 2, z),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y + 3, z)
		});
		break;
	case 2:
		return ::BSplineDeBoor(alphaZ, 5, std::vector<T>
		{
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y, z - 2),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y, z - 1),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y, z + 0),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y, z + 1),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y, z + 2),
			BSpline(level - 1, alphaX, alphaY, alphaZ, x, y, z + 3)
		});
		break;
	}
	throw "Impossible branch reached";
}

template<typename T>
inline T ScalarGrid<T>::HermiteConv(int level, T alphaX, T alphaY, T alphaZ, int x, int y, int z) const
{
	if (level > 2)
		level = 2;
	if (level <= 0)
	{
		return Hermite(alphaX,
			At(x - 1, y, z),
			At(x + 0, y, z),
			At(x + 1, y, z),
			At(x + 2, y, z));
	}
	else switch (level)
	{
	case 1:
		return Hermite(alphaY,
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y - 1, z),
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y + 0, z),
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y + 1, z),
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y + 2, z));
		break;
	case 2:
		return Hermite(alphaZ,
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y, z - 1),
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 0),
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 1),
			HermiteConv(level - 1, alphaX, alphaY, alphaZ, x, y, z + 2));
		break;
	}
	throw "Impossible branch reached";
}

template<typename T>
template<typename elemT>
inline bool ScalarGrid<T>::ReadFromBinary(const std::string& filename, uint sizeX, uint sizeY, uint sizeZ, bool bFlipY)
{
	// Failsafe 1
	if (sizeX == 0 || sizeY == 0 || sizeZ == 0)
	{
		Clear();
		return false;
	}

	std::ifstream inf = std::ifstream(filename, std::ios::out | std::ios::binary);

	// Failsafe 2
	if (!inf.is_open()) 
	{
		std::cerr << "File " << filename << " could not be open for reading" << std::endl;
		return false;
	}

	inf.seekg(0, inf.end);
	auto end_pos = inf.tellg();
	inf.seekg(0, inf.beg);

	// Failsafe 3
	if (uint(end_pos) < sizeX * sizeY * sizeZ * sizeof(T))
	{
		std::cerr << "Insufficient file size for the grid of size (" << sizeX << ", " << sizeY << ", " << sizeZ << ")" << std::endl;
		std::cerr << "Expected size: " << sizeX * sizeY * sizeZ * sizeof(T) << "; Received size: " << end_pos << std::endl;
		return false;
	}

	Resize(sizeX, sizeY, sizeZ);

	inf.read((char*)data.data(), sizeX * sizeY * sizeZ * sizeof(T));

	if (bFlipY) // This is only viable when sizeZ is 1
	{
#pragma omp parallel for
		for (int y = 0; y < sizeY / 2; ++y) 
			for (int x = 0; x < sizeX; ++x)
				swap(data[x + y * sizeX], data[x + (sizeY - 1 - y) * sizeX]);
	}

	return true;
}
template<typename T>
void ScalarGrid<T>::FlipY() 
{
	if (sizeX == 0 || sizeY == 0 || sizeZ == 0)
	{
		return;
	}
#pragma omp parallel for
	for (int y = 0; y < sizeY / 2; ++y)
		for (int x = 0; x < sizeX; ++x)
			swap(data[x + y * sizeX], data[x + (sizeY - 1 - y) * sizeX]);
}

