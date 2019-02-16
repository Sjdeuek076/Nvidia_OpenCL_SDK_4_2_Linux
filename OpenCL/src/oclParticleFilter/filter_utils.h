#ifndef filter_utils_h
#define filter_utils_h

#include <string>
#include <vector>
#include "recipes/nr.h"

using namespace std;

class vol_params 
{
private:
	double omega, 
		theta, 
		xi, 
		rho, 
		p, 
		mle, 
		chi2, 
		kalman_chi2, 
		adjusted_kalman_chi2;
	double *u, *v, *estimates;
public:
	double get_omega();
	double get_theta();
	double get_xi();
	double get_rho();
	double get_p();
	double get_mle();
	double get_chi2();
	double get_kalman_chi2();
	double get_adjusted_kalman_chi2();
	void set_omega(double omega_);
	void set_theta(double theta_);
	void set_xi(double xi_);
	void set_rho(double rho_);
	void set_p(double p_);
	void set_mle(double mle_);
	void set_chi2(double chi2_);
	void set_kalman_chi2(double kalman_chi2_);
	void set_adjusted_kalman_chi2(double adjusted_kalman_chi2_);
	void copy_params(vol_params& params);
	
	double* get_u();
	double* get_v();
	double* get_estimates();

	vol_params();
	
	vol_params(double omega_, 
		double theta_, 
		double xi_, 
		double rho_, 
		double p_, 
		double mle_, 
		double chi2_,
		double kalman_chi2_,
		double adjusted_kalman_chi2_);
	vol_params(const vol_params& copy); //copy constructor
	//~vol_params();
	/*
	Function that populates a vector with the initial parameters of the model
	*/
	void populate_starting_vector(Vec_IO_DP& starting_vector);
	void extract_params_from_vector(Vec_IO_DP& starting_vector);

	void set_best_estimate(const vol_params& params,
		const double* u,
		const double* v,
		const double* estimates,
		const int size);
};

void read_lines(string& fname, vector<double>& out);

/*
Builds the residuals using the classical definition, i.e. residual[i] = actual[i] - estimate[i];
*/
void build_classic_residuals(const vector<double>& actual, const double* log_estimate, double* residual);

/*
builds the residual as specified on page 95 of Alireza Javaheri's book.
i.e. residual[i] = (actual[i] - estimate[i]) / v[i];
*/
void build_kalman_residuals(const vector<double>& actual, const double* log_estimate, const double* v, double* residual);

/*
builds the residual as specified on page 95 of Alireza Javaheri's book, but subtracts the mean.
residual[i] = (actual[i] - estimate[i] - u[i]) / v[i];
*/
void build_mean_corrected_kalman_residuals(const vector<double>& actual, const double* log_estimate, const double* u, const double* v, double* residual);

/*
utility function for UKF
*/
int sqrt_matrix(double** pa, double** proda, int& N);

/**
Checks to see if the data is normal using the chi squared test with 20 degrees of freedom and alpha = 0.05
*/
bool is_normal(const double* data, bool normalize, int data_count, double& chi2_);

double cumulative_normal(const double x);

double gaussrand();

void normalize_data(const double* data_in, double* data_out, const int length);

// my codes
//double NormalCDFInverse(double p);
double NormalCDFInverse(long p);
double RationalApproximation(double t);
DP get_ran2(int &idum);
DP get_gammln(const DP xx);
void get_bessik(DP x, DP xnu, DP &ri, DP &rk, DP &rip, DP &rkp);
#endif
