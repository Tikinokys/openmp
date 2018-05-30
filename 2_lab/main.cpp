#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <random>
#include <omp.h>
#include <stdexcept>
#include <cmath>

double f(double x, double y){
	return 1;
}

double g(double x, double y){
	if(x==0)return 1-y*y;
	if(y==0)return 1-x*x;
	if(x==1)return 1-(y-1)*(y-1);
	return 1-(x-1)*(x-1);
}

std::vector<double>randVector(size_t size){
	std::vector<double> result(size);
	#pragma omp parallel shared(result)
	{
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::uniform_real_distribution<double> dis(1.0, 2.0);

		#pragma omp for schedule(static)
		for (size_t i=0; i<size; i++)
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

	std::string to_string() {
		std::stringstream ss;
		for(size_t i=0; i<(*this).rows(); ++i){
			for(size_t j=0; j<(*this).cols(); ++j){
				ss<<(*this)(i,j);
				if (j!=(*this).cols()-1)ss<<" ";
			}
			ss<<std::endl;
		}
		return ss.str();
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

struct Results_of_Dirichlet {
	size_t num_threads;
	Matrix surface;
	size_t iterations;
	double runtime;
	double eps;

	Results_of_Dirichlet(size_t nt, Matrix s, size_t i, double r, double e) : num_threads(nt), surface(s), iterations(i), runtime(r), eps(e) {}

	std::string to_string() {
		std::stringstream ss;
		ss << surface.to_string() << num_threads << "," << iterations << "," << runtime << "," << surface.cols() << "," << eps << std::endl;
		return ss.str();
	}
	std::string benchmark() {
		std::stringstream ss;
		ss << num_threads << "," << iterations << "," << runtime << "," << surface.cols() << "," << eps << std::endl;
		return ss.str();
	}
};

Matrix Solution_of_Dirichlet(size_t N,  double eps) {
	auto startTime = std::chrono::steady_clock::now();
	Matrix u_mat(N+2,  N+2);
	Matrix f_mat(N,  N);
	double h = 1.0 / (N + 1);

	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++)
			f_mat(i,  j) = f((i + 1) * h,  (j + 1) * h);
	}

	for (size_t i = 1; i < N + 1; i++) {
		u_mat(i,  0) = g(i * h,  0);
		u_mat(i,  N + 1) = g(i * h,  (N + 1) * h);
	}

	for (size_t j = 0; j < N + 2; j++) {
		u_mat(0,  j) = g(0,  j * h);
		u_mat(N + 1,  j) = g((N + 1) * h,  j * h);
	}

	double max;
	size_t iterations = 0;
	do {
		iterations++;
		max = 0;
		for (size_t i = 1; i < N + 1; i++)
			for (size_t j = 1; j < N + 1; j++) {
				double u0 = u_mat(i,  j);
				double t = 0.25 * (u_mat(i-1,  j) + u_mat(i+1,  j) + u_mat(i,  j-1) + u_mat(i,  j+1) - h*h*f_mat(i - 1,  j - 1));
				u_mat(i,  j) = t;
				double d = std::fabs(t - u0);
				if (d >	max) max = d;
			}
	} while (max > eps);

	auto runtime = std::chrono::steady_clock::now();
	auto runtimeDuration = std::chrono::duration_cast<std::chrono::duration<double>>(runtime - startTime);

	return u_mat;
}

Results_of_Dirichlet Solution_of_Dirichlet_OMP(size_t N,  double eps) {
	auto startTime = std::chrono::steady_clock::now();
	
	Matrix u_mat(N+2,  N+2);
	Matrix f_mat(N,  N);

	double h = 1.0 / (N + 1);

	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++)
			f_mat(i,  j) = f((i + 1) * h,  (j + 1) * h);
	}

	for (size_t i = 1; i < N + 1; i++) {
		u_mat(i,  0) = g(i * h,  0);
		u_mat(i,  N + 1) = g(i * h,  (N + 1) * h);
	}

	for (size_t j = 0; j < N + 2; j++) {
		u_mat(0,  j) = g(0,  j * h);
		u_mat(N + 1,  j) = g((N + 1) * h,  j * h);
	}

	double max, u0, d;
	size_t i = 0, j = 0, iterations = 0;
	std::vector<double> mx(N+2);
	do
	{
		iterations++;
		// нарастание волны (k - длина фронта волны)
		for (size_t k = 1; k < N+1; k++) {
			mx[k] = 0;
			#pragma omp parallel for shared(u_mat, k, mx) private(i, j, u0, d) schedule(static, 1)
			for (i = 1; i < k+1; i++) {
				j = k + 1 - i;
				u0 = u_mat(i, j);
				u_mat(i, j) = 0.25 * (u_mat(i-1, j) + u_mat(i+1, j) + u_mat(i, j-1) + u_mat(i, j+1) - h*h*f_mat(i-1, j-1));
				d = std::fabs(u_mat(i, j) - u0);
				if (d > mx[i]) mx[i] = d;
			}
		}
		for (size_t k = N-1; k > 0; k--) {
			#pragma omp parallel for shared(u_mat, k, mx) private(i, j, u0, d) schedule(static, 1)
			for (i = N-k+1; i < N+1; i++){
				j = 2*N - k - i + 1;
				u0 = u_mat(i, j);
				u_mat(i, j) = 0.25 * (u_mat(i-1, j) + u_mat(i+1, j) + u_mat(i, j-1) + u_mat(i, j+1) - h*h*f_mat(i-1, j-1));
				d = std::fabs(u_mat(i, j) - u0);
				if (d > mx[i]) mx[i] = d;				
			}
		}
		max = 0;
		for (i = 1; i < N+1; i++) {
			if (mx[i] > max) max = mx[i];
		}
	} while (max > eps);

	auto runtime = std::chrono::steady_clock::now();
	auto runtimeDuration = std::chrono::duration_cast<std::chrono::duration<double>>(runtime - startTime);

	Results_of_Dirichlet result(omp_get_max_threads(), u_mat, iterations, runtimeDuration.count(), eps);

	return result;
}

inline void print_CSV(int threads, size_t dim, double runtimeDuration) {
	std::cout << threads << "," << dim << "," << runtimeDuration << std::endl;
}

int main(int argc, char* argv[]){
	size_t N=99;
	double eps=0.001;
	if(argc>1){
		std::istringstream ss(argv[1]);
		int dim;
		if(!(ss>>dim)){
			throw std::invalid_argument("Invalid ARGV");
		}else{
			N=dim;
		}
	}
	std::cout<<Solution_of_Dirichlet_OMP(N-2,eps).benchmark();
	return 0;
}