/*
* CNDArray.h - Classes for dynamic allocating multidimensional arrays.
*
* Utilities of Vector, Matrix and 3D Matrix were adopted from nr3.h of
* Numercial Recipes 3rd Ed.
* Utilities of row-major and column-major conversion were provided by
* davekw7x@Numerical Recipes Forum
* As list in the following revision history, those original codes were either modified
* or extended.
*
* Maintained by Yixin Liu since 2010.4.17
* @Fudan Univ.
* Contact: liuyxpp@gmail.com
*/
/* $ Revision: 2010.5.5 $ */
/*
* Revision history
*	2010.4.17
*		1. Change the dimension length type and size type of NRvector, NRmatrix
*			and NRMat3d from int to size_t.
*		2. Add sizeByte() method to NRvector, NRmatrix and NRMat3d.
*		3. Add size() method to NRmatrix and NRMat3d.
*	2010.4.18
*		1. Add lyxMat4d class
*		2. lyxMat4d constructor, much clear version
*		3. lyxMat4d constructor, optimized version
*		4. Add getRaw() method to NRvector, NRmatrix, NRMat3d and lyxMat4d
*	2010.5.2
*		1. Add NRvector::sizeByte()
*	2010.5.5
*		1. Extracted from lyxUtil.h
*		2. splited into CNDArray.h and CNDArray.cpp
* 	2010.8.11
* 		1. Add definition of MatDoub etc. from NR.
* 	2010.10.31
* 		1. Add .nrows(),.ncols(),.nplanes() to NRMat3d.
*       2. Add NRMat3d::NRMat3d(size_t n,size_t m, size_t k, double a)
*       3. Add NRMat3d::operator =()
*
*/
#ifndef _CNDARRAY_H_
#define _CNDARRAY_H_

#include "lyxDef.h"

// all the system #include's we'll ever need
#include <cstdlib>
#include <cstring>
//#include <complex>
//#include "fftw3.h"

using namespace std;

//typedef complex<double> dcomplx;

template <class T>
class NRvector {
private:
	size_t nn;	// size of array. upper index is nn-1
	T *v;
public:
	NRvector();
	explicit NRvector(size_t n);		// Zero-based array
	NRvector(size_t n, const T &a);	//initialize to constant value
	NRvector(size_t n, const T *a);	// Initialize to array
	NRvector(const NRvector &rhs);	// Copy constructor
	NRvector & operator=(const NRvector &rhs);	//assignment
	typedef T value_type; // make T available externally
	T & operator[](const size_t i);	//i'th element
	const T & operator[](const size_t i) const;
	T* getRaw();
	const T* getRaw() const;
	size_t size() const;
	size_t sizeByte() const; // Total bytes occupied by the vector. Add by Yixin Liu, 2010.4.17
	void resize(size_t newn); // resize (contents not preserved)
	void assign(size_t newn, const T &a); // resize and assign a constant value
	~NRvector();
};

// NRvector definitions
template <class T>
inline T & NRvector<T>::operator[](const size_t i)	//subscripting
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn) {
		throw("NRvector subscript out of bounds");
	}
#endif
	return v[i];
}

template <class T>
inline const T & NRvector<T>::operator[](const size_t i) const	//subscripting
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn) {
		throw("NRvector subscript out of bounds");
	}
#endif
	return v[i];
}

template <class T>
inline T* NRvector<T>::getRaw()
{
	return v;
}

template <class T>
inline const T* NRvector<T>::getRaw() const
{
	return v;
}

template <class T>
inline size_t NRvector<T>::size() const
{
	return nn;
}

template <class T>
inline size_t NRvector<T>::sizeByte() const
{
	return nn*sizeof(T);
}
// end of NRvector definitions

template <class T>
class NRmatrix {
private:
	size_t nn;
	size_t mm;
	T **v;
public:
	NRmatrix();
	NRmatrix(size_t n, size_t m);			// Zero-based array
	NRmatrix(size_t n, size_t m, const T &a);	//Initialize to constant
	NRmatrix(size_t n, size_t m, const T *a);	// Initialize to array
	NRmatrix(const NRmatrix &rhs);		// Copy constructor
	NRmatrix & operator=(const NRmatrix &rhs);	//assignment
	typedef T value_type; // make T available externally
	T* operator[](const size_t i);	//subscripting: pointer to row i
	const T* operator[](const size_t i) const;
	T* getRaw();
	const T* getRaw() const;
	// transpose between row-major and column-major storage mode.
	void transpose(); // do transpose on itself
	void transpose(T* dest); // copy transpose to dest.
	void transpose(NRmatrix<T> &lhs); // copy transpose to another NRmatrix object.
	size_t nrows() const;
	size_t ncols() const;
	size_t size() const; // Number of elements. Add by Yixin Liu,2010.4.17.
	size_t sizeByte() const; // Total bytes occupied by the matrix. Add by Yixin Liu, 2010.4.17
	void resize(size_t newn, size_t newm); // resize (contents not preserved)
	void assign(size_t newn, size_t newm, const T &a); // resize and assign a constant value
	~NRmatrix();
};

template <class T>
inline T* NRmatrix<T>::operator[](const size_t i)	//subscripting: pointer to row i
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn) {
		throw("NRmatrix subscript out of bounds");
	}
#endif
	return v[i];
}

template <class T>
inline const T* NRmatrix<T>::operator[](const size_t i) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn) {
		throw("NRmatrix subscript out of bounds");
	}
#endif
	return v[i];
}

template <class T>
inline T* NRmatrix<T>::getRaw()
{
	return v[0];
}

template <class T>
inline const T* NRmatrix<T>::getRaw() const
{
	return v[0];
}

template <class T>
inline size_t NRmatrix<T>::nrows() const
{
	return nn;
}

template <class T>
inline size_t NRmatrix<T>::ncols() const
{
	return mm;
}

template <class T>
inline size_t NRmatrix<T>::size() const
{
	return nn*mm;
}

template <class T>
inline size_t NRmatrix<T>::sizeByte() const
{
	return nn*mm*sizeof(T);
}

template <class T>
class NRMat3d {
private:
	size_t nn;
	size_t mm;
	size_t kk;
	T ***v;
public:
	NRMat3d();
	NRMat3d(size_t n, size_t m, size_t k);
	NRMat3d(size_t n, size_t m, size_t k, const T &a);
	NRMat3d(size_t n, size_t m, size_t k, const T *a);
	NRMat3d & operator=(const NRMat3d &rhs);	//assignment
	T** operator[](const size_t i);	//subscripting: pointer to row i
	const T* const * operator[](const size_t i) const;
	T* getRaw();
	const T* getRaw() const;
	// transpose between row-major and column-major storage mode.
	void transpose(); // do transpose on itself
	void transpose(T* dest); // copy transpose to dest.
	void transpose(NRMat3d<T> &lhs); // copy transpose to another NRMat3d object.
	size_t dim1() const;
	size_t nplanes() const;
	size_t dim2() const;
	size_t nrows() const;
	size_t dim3() const;
	size_t ncols() const;
	size_t size() const;
	size_t sizeByte() const;
	void resize(size_t newn, size_t newm, size_t newk); // resize (contents not preserved)
	~NRMat3d();
};

template <class T>
inline T** NRMat3d<T>::operator[](const size_t i) //subscripting: pointer to row i
{
	return v[i];
}

template <class T>
inline const T* const * NRMat3d<T>::operator[](const size_t i) const
{
	return v[i];
}

template <class T>
inline T* NRMat3d<T>::getRaw()
{
	return v[0][0];
}

template <class T>
inline const T* NRMat3d<T>::getRaw() const
{
	return v[0][0];
}

template <class T>
inline size_t NRMat3d<T>::dim1() const
{
	return nn;
}

template <class T>
inline size_t NRMat3d<T>::nplanes() const
{
	return nn;
}

template <class T>
inline size_t NRMat3d<T>::dim2() const
{
	return mm;
}

template <class T>
inline size_t NRMat3d<T>::nrows() const
{
	return mm;
}

template <class T>
inline size_t NRMat3d<T>::dim3() const
{
	return kk;
}

template <class T>
inline size_t NRMat3d<T>::ncols() const
{
	return kk;
}

template <class T>
inline size_t NRMat3d<T>::size() const
{
	return nn*mm*kk;
}

template <class T>
inline size_t NRMat3d<T>::sizeByte() const
{
	return nn*mm*kk*sizeof(T);
}

template <class T>
class lyxMat4d {
/*
* lyxMat4d - Allocating 4D array with contiguous memory
*
* Devised by Yixin Liu since 2010.4.17
*/
private:
	size_t nn;
	size_t mm;
	size_t kk;
	size_t ww;
	T ****v;
public:
	lyxMat4d();
	lyxMat4d(size_t n, size_t m, size_t k, size_t w);
	T*** operator[](const size_t i);	//subscripting: pointer to row i
	const T* const * const * operator[](const size_t i) const;
	T* getRaw();
	const T* getRaw() const;
	// transpose between row-major and column-major storage mode.
	void transpose(); // do transpose on itself
	void transpose(T* dest); // copy transpose to dest.
	void transpose(lyxMat4d<T> &lhs); // copy transpose to another lyxMat4d object.
	size_t dim1() const;
	size_t dim2() const;
	size_t dim3() const;
	size_t dim4() const;
	size_t size() const;
	size_t sizeByte() const;
	void resize(size_t newn, size_t newm, size_t newk, size_t neww); // resize (contents not preserved)
	~lyxMat4d();
};

template <class T>
inline T*** lyxMat4d<T>::operator[](const size_t i) //subscripting: pointer to row i
{
	return v[i];
}

template <class T>
inline const T* const * const * lyxMat4d<T>::operator[](const size_t i) const
{
	return v[i];
}

template <class T>
inline T* lyxMat4d<T>::getRaw()
{
	return v[0][0][0];
}

template <class T>
inline const T* lyxMat4d<T>::getRaw() const
{
	return v[0][0][0];
}

template <class T>
inline size_t lyxMat4d<T>::dim1() const
{
	return nn;
}

template <class T>
inline size_t lyxMat4d<T>::dim2() const
{
	return mm;
}

template <class T>
inline size_t lyxMat4d<T>::dim3() const
{
	return kk;
}

template <class T>
inline size_t lyxMat4d<T>::dim4() const
{
	return ww;
}

template <class T>
inline size_t lyxMat4d<T>::size() const
{
	return nn*mm*kk*ww;
}

template <class T>
inline size_t lyxMat4d<T>::sizeByte() const
{
	return nn*mm*kk*ww*sizeof(T);
}

// vector types
typedef const NRvector<Int> VecInt_I;
typedef NRvector<Int> VecInt, VecInt_O, VecInt_IO;

typedef const NRvector<Uint> VecUint_I;
typedef NRvector<Uint> VecUint, VecUint_O, VecUint_IO;

typedef const NRvector<Llong> VecLlong_I;
typedef NRvector<Llong> VecLlong, VecLlong_O, VecLlong_IO;

typedef const NRvector<Ullong> VecUllong_I;
typedef NRvector<Ullong> VecUllong, VecUllong_O, VecUllong_IO;

typedef const NRvector<Char> VecChar_I;
typedef NRvector<Char> VecChar, VecChar_O, VecChar_IO;

typedef const NRvector<Char*> VecCharp_I;
typedef NRvector<Char*> VecCharp, VecCharp_O, VecCharp_IO;

typedef const NRvector<Uchar> VecUchar_I;
typedef NRvector<Uchar> VecUchar, VecUchar_O, VecUchar_IO;

typedef const NRvector<Doub> VecDoub_I;
typedef NRvector<Doub> VecDoub, VecDoub_O, VecDoub_IO;

typedef const NRvector<Doub*> VecDoubp_I;
typedef NRvector<Doub*> VecDoubp, VecDoubp_O, VecDoubp_IO;

typedef const NRvector<Complex> VecComplex_I;
typedef NRvector<Complex> VecComplex, VecComplex_O, VecComplex_IO;

typedef const NRvector<Bool> VecBool_I;
typedef NRvector<Bool> VecBool, VecBool_O, VecBool_IO;

// matrix types
typedef const NRmatrix<Int> MatInt_I;
typedef NRmatrix<Int> MatInt, MatInt_O, MatInt_IO;

typedef const NRmatrix<Uint> MatUint_I;
typedef NRmatrix<Uint> MatUint, MatUint_O, MatUint_IO;

typedef const NRmatrix<Llong> MatLlong_I;
typedef NRmatrix<Llong> MatLlong, MatLlong_O, MatLlong_IO;

typedef const NRmatrix<Ullong> MatUllong_I;
typedef NRmatrix<Ullong> MatUllong, MatUllong_O, MatUllong_IO;

typedef const NRmatrix<Char> MatChar_I;
typedef NRmatrix<Char> MatChar, MatChar_O, MatChar_IO;

typedef const NRmatrix<Uchar> MatUchar_I;
typedef NRmatrix<Uchar> MatUchar, MatUchar_O, MatUchar_IO;

typedef const NRmatrix<Doub> MatDoub_I;
typedef NRmatrix<Doub> MatDoub, MatDoub_O, MatDoub_IO;

typedef const NRmatrix<Bool> MatBool_I;
typedef NRmatrix<Bool> MatBool, MatBool_O, MatBool_IO;

// 3D matrix types
typedef const NRMat3d<Doub> Mat3dDoub_I;
typedef NRMat3d<Doub> Mat3dDoub, Mat3dDoub_O, Mat3dDoub_IO;

#endif /* _CNDARRAY_H_ */
