#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <random>
#include <omp.h>
#include <stdexcept>

std::vector<double> randVector(size_t size) {
	std::vector<double> result(size);
	#pragma omp parallel shared(result)
	{
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::uniform_real_distribution<double> dis(1.0, 2.0);

		#pragma omp for schedule(static)
		for (size_t i = 0; i < size; i++)
			result[i] = dis(gen);
	}
  return result;
}

class Matrix {
public:

	Matrix(size_t rows, size_t cols)
		: m_rows(rows)
		, m_cols(cols)
		, m_data(rows * cols)
	{}

	static Matrix rand(size_t rows, size_t cols) {
		return Matrix(rows, cols, randVector(rows*cols));
	}

	size_t rows() const {
		return m_rows;
	}

	size_t cols() const {
		return m_cols;
	}

	double& operator()(size_t row, size_t col) {
		return m_data[row * m_cols + col];
	}

	const double& operator()(size_t row, size_t col) const {
		return m_data[row * m_cols + col];
	}

	Matrix operator*(const Matrix& matrix);

private:
	size_t m_rows;
	size_t m_cols;
	std::vector<double> m_data;
	Matrix(size_t rows, size_t cols, std::vector<double>&& data)
		: m_rows(rows)
		, m_cols(cols)
		, m_data(std::move(data))
	{}
};


Matrix mulSerial(const Matrix& first, const Matrix& second) {
	Matrix result(first.rows(),second.cols());

	if (first.cols() == second.rows())
		for (size_t i = 0; i < result.rows(); ++i)
			for (size_t j = 0; j < result.cols(); ++j)
				for (size_t k = 0; k < result.rows(); ++k)
					result(i,j) += first(i, k) * second(k, j);
	else
		throw std::invalid_argument("Wrong dimensions");

	return result;
}

Matrix mulParallel(const Matrix& first, const Matrix& second) {
	Matrix result(first.rows(), second.cols());
	if (first.cols() == second.rows()) {
		#pragma omp parallel for shared(result, first, second)
		for (size_t i = 0; i < result.rows(); ++i) 
			for (size_t j = 0; j < result.cols(); ++j) {
				result(i, j) = 0;
				for (size_t k = 0; k < result.rows(); ++k) 
					result(i, j) += first(i, k) * second(k, j);
			}
	}
	else
		throw std::invalid_argument("Wrong dimensions");

	return result;
}

Matrix mulParallel2(const Matrix& first, const Matrix& second) {
	Matrix result(first.rows(), second.cols());
	if (first.cols() == second.rows()) {
		#pragma omp parallel for shared(result, first, second)
		for (size_t j = 0; j < result.cols(); ++j) 
			for (size_t i = 0; i < result.rows(); ++i) {
				result(i, j) = 0;
				for (size_t k = 0; k < result.rows(); ++k) 
					result(i, j) += first(i, k) * second(j, k);
			}
	}
	else
		throw std::invalid_argument("Wrong dimensions");

	return result;
}

Matrix Matrix::operator*(const Matrix& matrix) {
	return mulParallel2((*this), matrix);
}


inline void printCSV(int threads, size_t dim, double runtimeDuration) {
	std::cout << threads << "," << dim << "," << runtimeDuration << std::endl;
}



int main(int argc, char* argv[]) {

	auto startTime = std::chrono::steady_clock::now();

	size_t rows = 10;
	size_t cols = 10;

	if (argc > 1) {
		std::istringstream ss(argv[1]);
		int dim;
		ss >> dim;
		rows = cols = dim;
	}

	Matrix a = Matrix::rand(rows, cols);
	Matrix b = Matrix::rand(rows, cols);

	Matrix c = Matrix(rows, cols);
	c = a * b;

	auto mulTime = std::chrono::steady_clock::now();

	auto runtimeDuration = std::chrono::duration_cast<std::chrono::duration<double>>(mulTime - startTime);

	printCSV(omp_get_max_threads(), rows, runtimeDuration.count());

	return 0;
}