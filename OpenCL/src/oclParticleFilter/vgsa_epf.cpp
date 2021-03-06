// use NLopt library for nonlinear optimization library.
// http://ab-initio.mit.edu/wiki/index.php/NLopt
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <string>
#include "./recipes/nr.h"
#include "filters.h"
#include <vector>
#include "filter_utils.h"

using namespace std;

DP minimize_target_particle_extended_VGSA_parameters_bessel(Vec_I_DP & input);

int n_stock_prices = NULL;
//double log_stock_prices[n_stock_prices], u[n_stock_prices], v[n_stock_prices], estimates[n_stock_prices + 1];
double *log_stock_prices, *u, *v, *estimates , *ll, *errors;
//double muS = 0.687;
double muS = 0.0;
int call_counter = 0;
double lll = 0.0;

string input_file_name, output_file_name = "";
ofstream output_file;


int main(int argc, char** argv) {

	vector<double> prices;

	string usage = "syntax is program_name <input_file> <output_file>. "
		"The input file should be a 1 columned csv price file with a header, the output file is optional";
	//Parse the command line args.
	try {

		if(argc!=2 && argc!=3)
			throw usage;
		input_file_name = string(argv[1]);
		if(argc!=1)
			output_file_name = string(argv[2]);

		cout<<"input file is "<<input_file_name<<endl;

		if(!output_file_name.empty()) {
			cout<<"output_file is "<<output_file_name<<endl;
			output_file.open(output_file_name.c_str());
		}

		ifstream ifile(input_file_name.c_str());
		if(!ifile)
			throw "could not open input file " + input_file_name;

		ifile.close();
	} catch (string & exception) {
		cout<<exception<<endl;
		cout<<"please hit a key to continue..."<<endl;
		cin.get();
		exit(-1);
	}

	read_lines(input_file_name, prices);
	cout<<"Found "<<prices.size()<<" prices"<<endl;

	n_stock_prices = prices.size();

	log_stock_prices = new double[n_stock_prices];
	u = new double[n_stock_prices];
	v = new double[n_stock_prices];
	estimates = new double[n_stock_prices + 1];
    errors = new double[n_stock_prices +1];
    ll = &lll;

	for(int i = 0; i < prices.size(); i++) {
		log_stock_prices[i] = log(prices[i]);
	}

	//Initializing the starting point
	double a[6] = {0.1, 0.01, 0.1, 0.005, 0.02, 0.2};
	Vec_IO_DP starting_point(a, 6);

	//Initializing the identity matrix, don't know a more elegant
	//way to do it using the Numerical recipies api.
	Mat_IO_DP identity_matrix(4, 4);
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			if(i==j) identity_matrix[i][j] = 1.00;
			else identity_matrix[i][j] = 0.00;
		}
	}

	DP ftol = 1.00e-6;
	int iter;
	DP fret;

	NR::powell(starting_point, identity_matrix, ftol, iter, fret, minimize_target_particle_extended_VGSA_parameters_bessel);

	cout<<"Ran succesfully in "<<call_counter<<" iterations with return value "<<fret<<endl;
	double kappa = starting_point[0];
	double eta = starting_point[1];
	double lambda = starting_point[2];
	double sigma = starting_point[3];
    double theta = starting_point[4];
    double nu = starting_point[5];

	cout<<"Parameters are "<<" kappa = "<<kappa
			<<" eta = "<<eta
			<<" lambda = "<<lambda
            <<" sigma = "<<sigma
            <<" theta = "<<theta
			<<" nu = "<<nu<<endl<<endl;

	if(!output_file_name.empty())
		output_file.close();

	delete[] log_stock_prices, estimates, errors; // u, v, estimates;
	cout<<"Press any key to continue"<<endl;
	cin.get();
	return 0;
}

//DP minimize_target_particle_unscented_kalman_parameters_1_dim(Vec_I_DP & input) {
DP minimize_target_particle_extended_VGSA_parameters_bessel(Vec_I_DP & input){
	double kappa = input[0];
	double eta = input[1];
	double lambda = input[2];
	double sigma = input[3];
    double theta = input[4];
    double nu = input[5];

    estimate_particle_extended_VGSA_parameters_bessel(
													   log_stock_prices,
													   muS,
													   n_stock_prices,
													   kappa,
													   eta,
													   lambda,
													   sigma,
													   theta,
													   nu,
													   ll,
													   estimates,
													   errors);

/*
	double sum = 0;
	for(int i1 = 0; i1 < n_stock_prices; i1++)
		sum+=(log(v[i1])+u[i1]*u[i1]/v[i1]);
*/

	//print the header first time around
	if(call_counter==0 && !output_file_name.empty()) {
		output_file<<"iteration"<<","
			<<"kappa"<<","
			<<"eta"<<","
			<<"lambda"<<","
			<<"sigma"<<","
            <<"theta"<<","
            <<"nu"<<","
			<<"likelihood"<<endl;
	}

	output_file<<call_counter<<","
			<<kappa<<","
			<<eta<<","
			<<lambda<<","
			<<sigma<<","
			<<theta<<","
            <<nu<<","
            <<*ll<<endl;

	//Printing out a status message to see whats going on
	if(call_counter++ % 100 == 0)
		cout<<" call_counter = "<<call_counter
			<<" kappa = "<<kappa
			<<" eta = "<<eta
			<<" lambda = "<<lambda
			<<" sigma = "<<sigma
            <<" theta = "<<theta
            <<" nu = "<<nu
			<<" sum = "<<*ll //sum
			<<endl;

	return *ll;
}
