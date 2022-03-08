#pragma once

#include "Header.h"

#include <vector>
#include <memory>
#include <Geometry>



// Ray2D ListElement
struct ListElement 
{
	uint prev = list_elem_empty_flag;
	uint next = list_elem_empty_flag;
};


// Ray2D structure
// ----------------------------------------------------------------------------------------------------------

// State for integration ONLY
template<typename T>
struct Ray2D : public Eigen::Matrix<T, 4, 1> 
{
	// Type pseudonyms
	typedef class Eigen::Matrix<T, 4, 1> VecT; 
	typedef typename Eigen::Matrix<T, 2, 1> SubT;

	// Inhereting constructors/operators
	using VecT::VecT;

	// QotL constructors
	Ray2D() : VecT() {}
	Ray2D(const VecT& v) : VecT(v) {}
	Ray2D(const SubT& a, const SubT& b)
	{
		std::memcpy(this->data(), a.data(), 2 * sizeof(T));
		std::memcpy(this->data() + 2, b.data(), 2 * sizeof(T));
	}

	// QotL getters/setters
	operator VecT() { return static_cast<VecT>(*this); }

	SubT x() const { return SubT(this->data()[0], this->data()[1]); }
	SubT p() const { return SubT(this->data()[2], this->data()[3]); }

	void x(const SubT& other) { memcpy(this->data(), other.data(), 2 * sizeof(T)); }
	void p(const SubT& other) { memcpy(this->data() + 2, other.data(), 2 * sizeof(T)); }

};

// Ray3D structure
// ----------------------------------------------------------------------------------------------------------

// State for integration ONLY
template<typename T>
struct Ray3D : public Eigen::Matrix<T, 6, 1>
{
	// Type pseudonyms
	typedef class Eigen::Matrix<T, 6, 1> VecT;
	typedef typename Eigen::Matrix<T, 3, 1> SubT;

	// Inhereting constructors/operators
	using VecT::VecT;

	// QotL constructors
	Ray3D() : VecT() {}
	Ray3D(const VecT& v) : VecT(v) {}
	Ray3D(const SubT& a, const SubT& b)
	{
		std::memcpy(this->data(), a.data(), 3 * sizeof(T));
		std::memcpy(this->data() + 3, b.data(), 3 * sizeof(T));
	}

	// QotL getters/setters
	operator VecT() { return static_cast<VecT>(*this); }

	SubT x() const { return SubT(this->data()[0], this->data()[1], this->data()[2]); }
	SubT p() const { return SubT(this->data()[3], this->data()[4], this->data()[5]); }

	void x(const SubT& other) { memcpy(this->data(), other.data(), 3 * sizeof(T)); }
	void p(const SubT& other) { memcpy(this->data() + 3, other.data(), 3 * sizeof(T)); }

};

