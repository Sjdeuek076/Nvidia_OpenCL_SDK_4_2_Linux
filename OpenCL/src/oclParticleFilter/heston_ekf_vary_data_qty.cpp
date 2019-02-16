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

DP minimize_target_extended_kalman_parameters_1_dim(Vec_I_DP & input);

int n_stock_prices = 0;
double *log_stock_prices, *u, *v, *estimates;
double muS = 0.0;
int call_counter = 0;
DP ftol = 1.00e-6;

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
	
	//Initializing the starting point
	double a[4] = {0.2, 1.00, 0.5, -0.2};
	Vec_IO_DP starting_point(a, 4);
	
	//Initializing the identity matrix, don't know a more elegant
	//way to do it using the Numerical recipies api.
	Mat_IO_DP identity_matrix(4, 4);
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			if(i==j) identity_matrix[i][j] = 1.00;
			else identity_matrix[i][j] = 0.00;
		}
	}
	
	for(int i = 0; i < prices.size() - 200; i++) {
		
		n_stock_prices = prices.size() - i;

		cout<<"Number of prices being processed = "<<n_stock_prices<<endl;
		call_counter = 0;

		log_stock_prices = new double[n_stock_prices];
		u = new double[n_stock_prices];
		v = new double[n_stock_prices];
		estimates = new double[n_stock_prices + 1];

		for(int j = i; j < prices.size(); j++) {
			log_stock_prices[i] = log(prices[j]);
		}

		int iter;
		DP fret;

		NR::powell(starting_point, identity_matrix, ftol, iter, fret, minimize_target_extended_kalman_parameters_1_dim);
		
		double omega = starting_point[0];
		double theta = starting_point[1];
		double xi = starting_point[2];
		double rho = starting_point[3];
		
		//print the header first time around
		if(call_counter==0 && !output_file_name.empty()) {
			output_file<<"days"<<","
				<<"omega"<<","
				<<"theta"<<","
				<<"xi"<<","
				<<"rho"<<endl;
		}

		output_file<<call_counter<<","
			<<omega<<","
			<<theta<<","
			<<xi<<","
			<<rho<<endl;

		//clear resources
		delete [] log_stock_prices, u, v, estimates;
	}

	if(!output_file_name.empty())
		output_file.close();

	cout<<"Press any key to continue"<<endl;
	cin.get();
	return 0;
}

DP minimize_target_extended_kalman_parameters_1_dim(Vec_I_DP & input) {
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];

	estimate_extended_kalman_parameters_1_dim(log_stock_prices, 
		muS, 
		n_stock_prices, 
		omega, 
		theta, 
		xi, 
		rho, 
		u, 
		v, 
		estimates);

	double sum = 0;
	for(int i1 = 0; i1 < n_stock_prices; i1++)
		sum+=(log(v[i1])+u[i1]*u[i1]/v[i1]);
	
	//Printing out a status message to see whats going on
	if(call_counter++ % 100 == 0)
		cout<<" call_counter = "<<call_counter 
			<<" omega = "<<omega
			<<" theta = "<<theta
			<<" xi = "<<xi
			<<" rho = "<<rho
			<<" sum = "<<sum
			<<endl;

	return sum;
}
