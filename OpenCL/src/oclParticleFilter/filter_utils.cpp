#include "filter_utils.h"
#include "recipes/nr.h"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace NR;

void read_lines(string& fname, vector<double>& out) {
	std::ifstream fhandle(fname.c_str());
	char line[1000];
	int index = 0;
	do {
		fhandle.getline(line, 1000);
		if(!fhandle.eof())
			out.push_back(atof(line));

	} while(!fhandle.eof());

}

// Driver for numerical recipies choldc routine
//Cholesky to conform to the book.
int sqrt_matrix(double** pa, double** proda, int& N)
{
       
        int i,j;
        Vec_DP p(N),x(N);
        //Mat_DP a(*pa,N,N), atest(N,N), chol(N,N);
		Mat_DP a(N,N), atest(N,N), chol(N,N);
		
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				a[i][j] = pa[i][j];
			}
		}

		NR::choldc(a,p);
        for (i=0;i<N;i++) {
          for (j=0;j<N;j++) {
            if (i == j) chol[i][i]=p[i];
            else chol[i][j]=(i > j ? a[i][j] : 0.0);
			proda[i][j] = chol[i][j];
          }
        }
		return 0;
}

/**
Checks to see if the data is normal using the k-s test
*/
bool is_normal(const double* data, bool normalize, int data_count, double& chi2_)
{
	const int bin_count = 22;
	const int edge_count = bin_count - 1;
	double bin_edges[edge_count];
	const double starting_bin = -2.5;
	const double increment = 0.25;
	
	cout.setf(ios::fixed);
	cout<<setprecision(10);

	//ebins - expected bins, bins - actual bins
	Vec_IO_DP ebins(bin_count), bins(bin_count);
	for(int i = 0; i < edge_count; i++) 
	{
		bin_edges[i] = starting_bin + increment * ((double) i);
		if(i==0)  //The first bin
		{			
			ebins[i] = ((double) data_count) * cumulative_normal(bin_edges[i]);
		}
		else
		{
			ebins[i] = ((double) data_count) * (cumulative_normal(bin_edges[i]) - cumulative_normal(bin_edges[i-1]));
		} 

		bins[i] = 0.00; //initialing to zero.
	}

	//the last ebin: ebin[bin_count - 1] =  is (1 - cumulative_normal(bin_edges[edge_count - 1])) * data_count
	ebins[bin_count - 1] = (1 - cumulative_normal(bin_edges[edge_count - 1])) * data_count;
	bins[bin_count - 1] = 0;

	double* normalized_data = new double[data_count];
        if(normalize)
        {
         	normalize_data(data, normalized_data, data_count);
	}
	else //if we aren't normalizing the data, just copy it.
	{
		for(int i = 0; i < data_count; i++)
		{
			normalized_data[i] = data[i];
		}
    }


	//binning the data
	for(int i = 0; i < data_count; i++)
	{
		bool binned = false;
		double num = normalized_data[i];
		//iterate thru the edges looking for the appropriate bin to bin it into
		for(int j = 0; j < edge_count; j++)
		{
			if(num <= bin_edges[j]) {
				binned = true;
				bins[j]++;
				break;
			} 
		} 
		if(!binned)
		{
			bins[edge_count]++;
		}
	}

	double df, chsq, prob;

	NR::chsone(bins, ebins, 2, df, chsq, prob);
	cout<<"df = "<<df<<endl
		<<"chsq = "<<chsq<<endl
		<<"prob = "<<prob<<endl;
	
	delete[] normalized_data;

	chi2_ = chsq;

	return chsq < 31.50;
}

double cumulative_normal(const double x_)
{
	
    //return 1.0 - NR::erfcc(x_/sqrt(2.0));

	double x = x_;
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}

void normalize_data(const double* data_in, double* data_out, const int length)
{
	//get the moments of the data for normalization
	Vec_I_DP data_vecor(data_in, length);
	double ave, adev, sdev, var, skew, curt;
	NR::moment(data_vecor, ave, adev, sdev, var, skew, curt);
	for(int i = 0; i < length; i++) {
		data_out[i] = (data_in[i] - ave) / sdev;
	}
}

double gaussrand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if(phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}

/*
Builds the residuals using the classical definition, i.e. residual[i] = actual[i] - estimate[i];
*/
void build_classic_residuals(const vector<double>& actual, const double* log_estimate, double* residual)
{
	for(unsigned int i = 0; i < actual.size(); i++) 
	{
		residual[i] = actual[i] - exp(log_estimate[i]);
	}
}

/*
builds the residual as specified on page 95 of Alireza Javaheri's book.
i.e. residual[i] = (actual[i] - estimate[i]) / v[i];
*/
void build_kalman_residuals(const vector<double>& actual, const double* log_estimate, const double* v, double* residual)
{
	for(unsigned int i = 0; i < actual.size(); i++)
	{
		residual[i] = (actual[i] - exp(log_estimate[i])) / v[i];
	}
}

/*
builds the residual as specified on page 95 of Alireza Javaheri's book, but subtracts the mean.
residual[i] = (actual[i] - estimate[i] - u[i]) / v[i];
*/
void build_mean_corrected_kalman_residuals(const vector<double>& actual, const double* log_estimate, const double* u, const double* v, double* residual)
{
	for(unsigned int i = 0; i < actual.size(); i++)
	{
		residual[i] = (actual[i] - exp(log_estimate[i]) - u[i]) / v[i];
	}
}



/********************************
From here until the end of the file, the methods of vol_params are implemented.
********************************/

double vol_params::get_omega()
{
	return omega;
}
	
double vol_params::get_theta()
{
	return theta;
}

double vol_params::get_xi()
{
	return xi;
}

double vol_params::get_rho()
{
	return rho;
}

double vol_params::get_p()
{
	return p;
}

double vol_params::get_mle()
{
	return mle;
}

double vol_params::get_kalman_chi2()
{
	return kalman_chi2;
}

double vol_params::get_adjusted_kalman_chi2()
{
	return adjusted_kalman_chi2;
}

double vol_params::get_chi2()
{
	return chi2;
}
	
void vol_params::set_omega(double omega_)
{
	omega = omega_;
}
	
void vol_params::set_theta(double theta_)
{
	theta = theta_;
}
	
void vol_params::set_xi(double xi_)
{
	xi = xi_;
}
	
void vol_params::set_rho(double rho_)
{
	rho = rho_;
}

void vol_params::set_p(double p_) 
{
	p = p_;
}

void vol_params::set_mle(double mle_)
{
	mle = mle_;
}

void vol_params::set_chi2(double chi2_)
{
	chi2 = chi2_;
}

void vol_params::set_kalman_chi2(double kalman_chi2_)
{
	kalman_chi2 = kalman_chi2_;
}

void vol_params::set_adjusted_kalman_chi2(double adjusted_kalman_chi2_)
{
	adjusted_kalman_chi2 = adjusted_kalman_chi2_;
}

void vol_params::copy_params(vol_params& params)
{
	omega = params.omega;
	p = params.p;
	rho = params.rho;
	theta = params.theta;
	xi = params.xi;
}

vol_params::vol_params()
{
	u = NULL;
	v = NULL;
	estimates = NULL;

	mle = pow(10.00, 6);
	chi2 = pow(10.00, 6);
	omega = 0.00;
	p = 0.00;
	rho = 0.00;
	theta = 0.00;
	xi = 0.00;

}

/*vol_params::~vol_params(void)
{
	cout<< "object vol_params is being deleted !"<<endl;
}*/

vol_params::vol_params(double omega_, 
	double theta_, 
	double xi_, 
	double rho_, 
	double p_, 
	double mle_, 
	double chi2_,
	double kalman_chi2_,
	double adjusted_kalman_chi2_)
{
	u = NULL;
	v = NULL;
	estimates = NULL;

	mle = mle_;
	chi2 = chi2_;
	omega = omega_;
	p = p_;
	rho = rho_;
	theta = theta_;
	xi = xi_;
	kalman_chi2 = kalman_chi2_;
	adjusted_kalman_chi2 = adjusted_kalman_chi2_;
}

//copy constructor
vol_params::vol_params(const vol_params& copy) 
{
	u = NULL;
	v = NULL;
	estimates = NULL;

	mle = copy.mle;
	chi2 = copy.chi2;
	kalman_chi2 = copy.chi2;
	adjusted_kalman_chi2 = copy.adjusted_kalman_chi2;

	omega = copy.omega;
	p = copy.p;
	rho = copy.rho;
	theta = copy.theta;
	xi = copy.xi;
}

void vol_params::populate_starting_vector(Vec_IO_DP& starting_vector)
{
	starting_vector[0] = omega;
	starting_vector[1] = theta;
	starting_vector[2] = xi;
	starting_vector[3] = rho;
	if(starting_vector.size() == 5)
		starting_vector[4] = p;
}

void vol_params::extract_params_from_vector(Vec_IO_DP& v)
{
	omega = v[0];
	theta = v[1];
	xi = v[2];
	rho = v[3];
	if(v.size() == 5)
		p = v[4];
}

void vol_params::set_best_estimate(const vol_params& params,
		const double* u_,
		const double* v_,
		const double* estimates_,
		const int size) 
{
	mle = params.mle;
	chi2 = params.chi2;
	kalman_chi2 = params.kalman_chi2;
	adjusted_kalman_chi2 = params.adjusted_kalman_chi2;

	if(u == NULL) {
		u = new double[size];
		v = new double[size];
		estimates = new double[size];
	}
	
	for(int i = 0; i < size; i++)
	{
		u[i] = u_[i];
		v[i] = v_[i];
		estimates[i] = estimates_[i];
	}

	omega = params.omega;
	p = params.p;
	rho = params.rho;
	theta = params.theta;
	xi = params.xi;
}

double* vol_params::get_u()
{
	return u;
}

double* vol_params::get_v()
{
	return v;
}

double* vol_params::get_estimates()
{
	return estimates;
}

// my codes
//DP get_ran2(long &idum)
DP get_ran2(int &idum)
{
   return NR::ran2(idum);
}

DP get_gammln(const DP xx)
{
    return NR::gammln(xx);
}

void get_bessik(DP x, DP xnu, DP &ri, DP &rk, DP &rip, DP &rkp)
{
    NR::bessik(x, xnu, ri, rk, rip, rkp);

}
