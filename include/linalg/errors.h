#ifndef LINALG_ERRORS_H
#define LINALG_ERRORS_H

#include <stdexcept>

namespace linalg {

//==============================================================================
// Library Root Level Error
//==============================================================================
class LinalgError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};

//==============================================================================
// Mathematical Property Errors
//==============================================================================
class PropertyError : public LinalgError
{
public:
    using LinalgError::LinalgError;
};

class Singular : public PropertyError
{
public:
    using PropertyError::PropertyError;
};

class Indefinite : public PropertyError
{
public:
    using PropertyError::PropertyError;
};

class NonSymmetric : public PropertyError
{
public:
    using PropertyError::PropertyError;
};

class NonLowerTriangular : public PropertyError
{
public:
    using PropertyError::PropertyError;
};

class NonUpperTriangular : public PropertyError
{
public:
    using PropertyError::PropertyError;
};

//==============================================================================
// Shape Errors
//==============================================================================
class ShapeError : public LinalgError
{
public:
    using LinalgError::LinalgError;
};

}

#endif // LINALG_ERRORS_H