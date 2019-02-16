#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include "./recipes/nr.h"
#include "filters.h"
#include <vector>
#include "filter_utils.h"
#include <ctime>

/* 
	0.1 string get_VolParams_string
	0.2 VolParams get_next_VolParam_to_preturb
1. void g
	1.1 void init_log
2. DP minimize_target_ekf
	2.1 void log_parameters
3. void mark_better_parameters
4. void log_stats
*/ 

using namespace std;

enum VolModel {HESTON = 1, GARCH = 2, THREE_TWO = 3, VAR_P = 4};
enum RunMode {SIMPLE = 1, NORMAL_RESIDUALS = 2, UNCORRELATED_RESIDUALS = 3, NORMAL_AND_UNCORRELATED_RESIDUALS = 4};
enum FilterType {EKF = 1, UKF = 2};
enum VolParams {OMEGA, THETA, XI, ROE, P};

const double rand_max = RAND_MAX;
int n_stock_prices = 0;
double *log_stock_prices, *u, *v, *estimates;
double muS = 0.0;

double p_heston = 0.50;
double p_garch  = 1.00;
double p_3_2    = 1.50;
double p_variable = 1; //This is the starting case situation where we try to optimize for variable p
int max_simulations = 1;
VolModel model;
RunMode runmode;
FilterType filtertype;

int call_counter = 0;
int simulation_counter = 1;

ofstream paramter_file;
ofstream residual_file;
ofstream log_file;


DP ftol = 1.00e-5;  //

//------------- function declaration 

void init_log(const VolModel model, const RunMode runmode, ofstream& log_stream);
string get_run_mode(const RunMode runmode);
string get_model_name(const VolModel model);
string get_filter_type(const FilterType filtertype);

/*void ekf_filter(vector<double>& prices,
	    string& parameter_file_name,
		string& residual_file_name);*/

void estimate_filter(vector<double>& prices,
		string& parameter_file_name,
		string& residual_file_name);

void log_better_param(ofstream& log_file,        
		vol_params& best_params,
		vol_params& current_params,
		//int& simulation_counter,
		vector<double>& prices);
	    
void log_residual(ofstream& residual_file,
        vol_params& best_params,
        vol_params& current_params,
		vector<double>& prices,
		double* classic_residuals,
	    double* kalman_residuals, 
	    double* corrected_kalman_residuals);	  

void parse_args(int argc, char** argv, 
	string& input_file_name,
    string& parameter_file_name,
    string& residual_file_name);

void log_stats(ofstream& lf,
	const double* classic_residuals,
	const double* kalman_residuals,
	const double* kalman_residuals_corrected,
	const int count);

DP minimize_target_filter(Vec_I_DP & input);

void log_parameters(double& omega,
	double& theta,
	double& xi,
	double& rho,
	double& p,
	double& mle);

void mark_better_parameters(int& simulation_counter,
	int& max_simulations,
	vol_params& current_best_params,
	vol_params& new_best_params,
	ofstream& log_stream,
	int price_count);

string get_VolParams_string(VolParams& vp);

//Gets the next vol parameter to preturb
/*
@param last_param the last param that was used to preturb the solution.
@param result_improved whether the last preturbation resulted in an improved result.
@param current_model: The current vol model being used.
*/
VolParams get_next_VolParam_to_preturb(VolParams last_param, bool& result_improved, VolModel& current_model);

int main(int argc, char** argv) {
	
	srand(time(NULL)); //seed the random number generator

	vector<double> prices;
	string input_file_name, parameter_file_name = "", residual_file_name = "";
	
	parse_args(argc, argv,
		input_file_name,
		parameter_file_name,
		residual_file_name);

	//setting cout to show 10 decimal places by default
	std::cout.setf(ios::fixed);
	std::cout<<setprecision(10); 
	read_lines(input_file_name, prices);
	cout<<"Found "<<prices.size()<<" prices"<<endl;
	
	n_stock_prices = prices.size();
	
	log_stock_prices = new double[n_stock_prices];
	for(unsigned int i = 0; i < prices.size(); i++) {
		log_stock_prices[i] = log(prices[i]);
	}

	estimate_filter(prices, parameter_file_name, residual_file_name);
	
	delete[] log_stock_prices;
	return 0;
}

void estimate_filter(vector<double>& prices,
		string& parameter_file_name,
		string& residual_file_name)
{
    n_stock_prices = prices.size();
	u = new double[n_stock_prices];
	v = new double[n_stock_prices];
	estimates = new double[n_stock_prices + 1];
	/*param variables are
	 * self define class; defination in filter_utils.h
	 */ 
	vol_params current_params(  //initial constructor 
	    0.02, //omega
		1.00,  //theta
		0.50, //xi
		-0.20, //rho
		1.00,  //p
		pow(10.00, 6), //mle
		pow(10.00, 6), //chi2
		pow(10.00, 6), //kalman_chi2
		pow(10.00, 6) //(mean) adjusted_kalman_chi2 
		); 
	vol_params best_params(current_params); //initial constructor
	
	Vec_IO_DP* start_; //files in nrtypes_lib.h self define type 
	
	Mat_IO_DP* identity_matrix; //files in nrtypes_lib.h self define type 

	if(model!=VAR_P) //if we aren't trying to estimate p, we only need a 4x4 matrix
	{
		//Initializing the starting point
		start_ = new Vec_IO_DP(4);
		identity_matrix = new Mat_IO_DP(4, 4);
	} else { //we are trying to estimate p, so we want a 5x5 matrix
		//Initializing the starting point
		start_ = new Vec_IO_DP(5);
		identity_matrix = new Mat_IO_DP(5, 5);
	}

	Vec_IO_DP& start = *start_; //aliasing the pointer for easy reference
	
	paramter_file.open(parameter_file_name.c_str());
	residual_file.open(residual_file_name.c_str());
	//setting the output to 10 decimal places
	paramter_file.setf(ios::fixed);
	paramter_file<<setprecision(10);
	residual_file.setf(ios::fixed);
	residual_file<<setprecision(10);

	double* classic_residuals = new double[n_stock_prices];
	double* kalman_residuals = new double[n_stock_prices];
	double* corrected_kalman_residuals = new double[n_stock_prices];

	double chi2; //buffer variable.
	
	while(simulation_counter <= max_simulations)
	{
		int nvars = (model!=VAR_P) ? 4 : 5; 
		for(int i = 0; i < nvars; i++) 
		{
			for(int j = 0; j < nvars; j++)
			{
				(*identity_matrix)[i][j] = (i==j) ? 1 : 0;
			}
		}

		current_params.populate_starting_vector(start);
		for(int i = 0; i < 5; i++)
			cout<<start[i]<<endl;
		int iter;
		DP mle = 0.00;
		
		NR::powell(start, *identity_matrix, ftol, iter, mle, minimize_target_filter);
		
		if(model==VAR_P)
			cout<<" p = "<<current_params.get_p();
		cout<<endl<<endl;

		/*
		* Setting the current params out of the optimizations.
		*/
		current_params.extract_params_from_vector(start);
		current_params.set_mle(mle);
		
		//build_classic_xx are definded in filter_utils.cpp
		build_classic_residuals(prices, estimates, classic_residuals);
        build_kalman_residuals(prices, estimates, v, kalman_residuals);
        build_mean_corrected_kalman_residuals(prices, estimates, u, v, corrected_kalman_residuals);
		//getting the chi2 statistic for the classic residuals
		is_normal(classic_residuals, true, prices.size(), chi2); current_params.set_chi2(chi2);
		is_normal(kalman_residuals, true, prices.size(), chi2); current_params.set_kalman_chi2(chi2);
		is_normal(corrected_kalman_residuals, true, prices.size(), chi2); current_params.set_adjusted_kalman_chi2(chi2);

		if(runmode == NORMAL_RESIDUALS) 
		// my code, log_file
		log_better_param(log_file,
			best_params, 
			current_params, 
			prices);
						
		simulation_counter++;
		paramter_file.flush(); //flush the buffer to disc		
	}
	
	log_residual(residual_file, 
		best_params, 
		current_params,
		prices, 
		classic_residuals,  
		kalman_residuals,  
		corrected_kalman_residuals);

    //the ending of the log_file
	log_stats(log_file,
		classic_residuals,
		kalman_residuals,
		corrected_kalman_residuals,
		prices.size());

	residual_file.close();
	paramter_file.close();

	delete[] u, v, estimates, classic_residuals, kalman_residuals, corrected_kalman_residuals;
	delete start_, identity_matrix;
}



void log_better_param(ofstream& log_file,        
		vol_params& best_params,
		vol_params& current_params,
		//int& simulation_counter,
		vector<double>& prices)
{
	{
			bool solution_improved = false;

			//if(best_params.get_mle() > mle)a
			if(best_params.get_kalman_chi2() > current_params.get_kalman_chi2())
			//if(best_params.get_mle() > mle && best_params.get_kalman_chi2() > current_params.get_kalman_chi2())
			{
				solution_improved = true;
				mark_better_parameters(simulation_counter, 
					max_simulations, 
					best_params, 
					current_params, 
					log_file, 
					prices.size());
			} 
			else 
			{
				log_file<<"We have NOT found better parameters on simulation # "<<simulation_counter << " out of "<< max_simulations <<endl
					<<"-- new mle                       = "<< current_params.get_mle()  <<" old = "<<best_params.get_mle()<<endl
					<<"-- new chi2                      = "<<current_params.get_chi2() <<" old = "<<best_params.get_chi2()<<endl
					<<"-- new kalman chi2               = "<<current_params.get_kalman_chi2()<<" old = "<<best_params.get_kalman_chi2()<<endl
					<<"-- new mean adjusted kalman chi2 = "<<current_params.get_adjusted_kalman_chi2()<<" old = "<<best_params.get_adjusted_kalman_chi2()<<endl
					<<"-- Resetting the parameters to the previous best estimate, and then preturbing..."<<endl;
				current_params.copy_params(best_params);
			}
				
			current_params.set_omega(abs(gaussrand() * sqrt(2.00) + best_params.get_omega()));
			current_params.set_theta(abs(gaussrand() * sqrt(2.00) + best_params.get_theta()));
			current_params.set_xi(abs(gaussrand() * sqrt(2.00) + best_params.get_xi()));
			current_params.set_rho(((rand()/rand_max) * 2.00) - 1.00);
			current_params.set_p(abs(((double)rand())/rand_max) + 0.5); 

			log_file<<"old omega "<<best_params.get_omega()
				<<" new -> "<<current_params.get_omega()
				<<" diff= "<<best_params.get_omega() - current_params.get_omega()<<endl;
			log_file<<"old theta "<<best_params.get_theta()
				<<" new-> "<<current_params.get_theta()
				<<" diff= "<<best_params.get_theta() - current_params.get_theta()<<endl;
			log_file<<"old xi    "<<best_params.get_xi()   
				<<" new-> "<<current_params.get_xi()
				<<" diff= "<<best_params.get_xi() - current_params.get_xi()<<endl;
			log_file<<"old rho   "<<best_params.get_rho()  
				<<" new-> "<<current_params.get_rho()
				<<" diff= "<<best_params.get_rho() - current_params.get_rho()<<endl;
			if(model==VAR_P)
				log_file<<"old p     "<<best_params.get_p()
					<<" new-> "<<current_params.get_p()<<" diff= "
					<<best_params.get_p() - current_params.get_p()<<endl;

			log_file<<endl<<endl<<endl;
			call_counter = 0;
		} 			
}

void log_residual(ofstream& residual_file,
        vol_params& best_params,
        vol_params& current_params,
		//int& simulation_counter,
		vector<double>& prices,
		double* classic_residuals,
	    double* kalman_residuals, 
	    double* corrected_kalman_residuals)
{
	residual_file<<"LineNum"<<","
		<<"Prices"<<","
		<<"Estimates"<<","
		<<"ClassicResidual"<<","
		<<"KalmanResidual"<<","
		<<"CorrectedKalmanResidual"<<endl;
	if(runmode==SIMPLE) {
		log_file<<"Optimum values are "<<endl
			<<"--omega "<<current_params.get_omega()<<endl
			<<"--theta "<<current_params.get_theta()<<endl
			<<"--rho   "<<current_params.get_rho()<<endl
			<<"--xi    "<<current_params.get_xi()<<endl;
		if(model==VAR_P)
			log_file<<"--p     "<<current_params.get_p()<<endl;
		for(unsigned int i = 0; i < prices.size(); i++) {
			residual_file<<i<<","
                <<prices[i]<<","
                <<exp(estimates[i])<<","
				<<classic_residuals[i]<<","
				<<kalman_residuals[i]<<","
				<<corrected_kalman_residuals[i]<<endl;
		}
	} else {
		//need to build the residuals again.
		build_classic_residuals(prices, 
			best_params.get_estimates(), 
			classic_residuals);
        build_kalman_residuals(prices, 
			best_params.get_estimates(), 
			best_params.get_v(), 
			kalman_residuals);
        build_mean_corrected_kalman_residuals(prices, 
			best_params.get_estimates(), 
			best_params.get_u(), 
			best_params.get_v(), 
			corrected_kalman_residuals);

		log_file<<"Optimum values are "<<endl
			<<"--omega "<<best_params.get_omega()<<endl
			<<"--theta "<<best_params.get_theta()<<endl
			<<"--rho   "<<best_params.get_rho()<<endl
			<<"--xi    "<<best_params.get_xi()<<endl;
		if(model==VAR_P)
			log_file<<"--p     "<<best_params.get_p()<<endl; //only for VAR_P(=4) not p value inputed
		for(unsigned int i = 0; i < prices.size(); i++) {
			residual_file<<i<<","
                <<prices[i]<<","
			    <<exp(estimates[i])<<","
			    <<classic_residuals[i]<<","
                <<kalman_residuals[i]<<","
                <<corrected_kalman_residuals[i]<<endl;
		}
	}	
}


void log_stats(ofstream& lf,
        const double* classic_residuals,
        const double* kalman_residuals,
        const double* kalman_residuals_corrected,
        const int count)
{
	double classic_chi2, kalman_chi2, kalman_corrected_chi2,
		classic_chi2_normalized, kalman_chi2_normalized, kalman_corrected_chi2_normalized;
	

	is_normal(classic_residuals, true, count, classic_chi2_normalized);
	is_normal(classic_residuals, false, count, classic_chi2);

	is_normal(kalman_residuals, true, count, kalman_chi2_normalized);
        is_normal(kalman_residuals, false, count, kalman_chi2);

	is_normal(kalman_residuals_corrected, true, count, kalman_corrected_chi2_normalized);
	is_normal(kalman_residuals_corrected, false, count, kalman_corrected_chi2);

	lf << " classic_chi2                      "<<classic_chi2<<"\n"
		<< " classic_chi2_normalied            "<<classic_chi2_normalized<<"\n"
		<< " kalman_chi2                       "<<kalman_chi2<<"\n"
		<< " kalman_chi2_normalized            "<<kalman_chi2_normalized<<"\n"
		<< " kalman_corrected_chi2             "<<kalman_corrected_chi2<<"\n"
        << " kalman_corrected_chi2_normalized  "<<kalman_corrected_chi2_normalized<<endl;	
}


void parse_args(int argc, char** argv,
	string& input_file_name,
    string& parameter_file_name,
    string& residual_file_name) {

	string usage = "syntax is program_name <input_file> <parameter_output_file> <residual_output_file> "
		"<MODEL 1=HESTON, 2=GARCH, 3=3_2, 4=variable p> "
		"<RUNMODE 1=SIMPLE, 2=NORMAL_RESIDUALS>. "
        "<FILTERTYPE 1=EKF, 2=UKF>. "
		"<MAXSIMULATIONS: Required if RUNMODE!=1> \n "
		"The input file should be a 1 columned csv price file, the output file(s) will be created";

	//Parse the command line args.
	try {
		if(argc < 7) 
			throw usage;
		input_file_name = string(argv[1]);
		
		parameter_file_name = string(argv[2]);
		residual_file_name = string(argv[3]);
		
		string model_ = string(argv[4]);
		string runmode_ = string(argv[5]);
		string filtertype_ = string(argv[6]);
		if(argc==8) 
		{
			max_simulations = atoi(argv[7]);
			cout<<"max_simulations is set to "<<max_simulations<<endl;
		}

		std::cout<<"input file is "<<input_file_name<<endl;
		
		//std::cout<<"parameter_output_file is "<<parameter_file_name<<endl;
		//std::cout<<"residual_output_file is "<<residual_file_name<<endl;

		if(model_!="1" &&
			model_!="2" &&
			model_!="3" &&
			model_!="4")
			throw "model is not one of 1,2,3,4. Usage is \n" + usage;

		if(runmode_!="1" &&
			runmode_!="2")
			throw "runmode is not one of 1,2. Usage is \n" + usage;

		if(runmode_!="1" && argc!=8)
			throw "RUNMODE!=1 and MAXSIMULATIONS not provided";

		//convert them into int's
		model = (VolModel) atoi(model_.c_str()); 
		runmode = (RunMode) atoi(runmode_.c_str());
		filtertype = (FilterType) atoi(filtertype_.c_str());
				
		ifstream ifile(input_file_name.c_str());
		if(!ifile)
			throw "could not open input file " + input_file_name;

		ifile.close();

		init_log(model, runmode, log_file);
	} catch (string& exception) {
		std::cout<<"CLI Parsing error"<<endl<<exception<<endl;
		std::cout<<"please hit a enter to continue..."<<endl;
		cin.get();
		exit(-1);
	}
}




DP minimize_target_filter(Vec_I_DP & input) 
{
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];
	double p;
	switch(model) 
	{
	case HESTON: p = p_heston; break;
	case GARCH: p = p_garch; break;
	case THREE_TWO: p = p_3_2; break;
	default: p = input[4];
	}
	
	switch(filtertype)
	{
	case EKF:
	estimate_ekf_parm_1_dim(log_stock_prices, 
		muS, 
		n_stock_prices, 
		omega, 
		theta, 
		xi, 
		rho, 
		p, //p as specified by the model
		u, 
		v, 
		estimates);
	break;
	
	case UKF:
	estimate_unscented_kalman_parameters_1_dim(log_stock_prices,
		muS, 
		n_stock_prices, 
		omega, 
		theta, 
		xi, 
		rho,
        p, 
		u, 
		v, 
		estimates);
	break;
    }
	double mle = 0.00;

	for(int i1 = 0; i1 < n_stock_prices; i1++)
		mle+=(log(v[i1])+u[i1]*u[i1]/v[i1]);

	//test for nan
	if(mle!=mle || 
		rho > 0.95 || rho < -0.95
		|| omega < 0
		|| xi < 0
		|| theta < 0
		|| p < 0.25 || p > 2)
		mle = pow(10.00, 5.00) + rand(); //add a random number to it so it doesn't settle there

	log_parameters(omega, theta, xi, rho, p, mle);

	return mle;
}

void log_parameters(double& omega,
	double& theta,
	double& xi,
	double& rho,
	double& p,
	double& mle) {

	//print the header first time around
	//if(call_counter==0) {
    if(simulation_counter==1 && call_counter==0){ 
		paramter_file
			<<"simulation"<<","
			<<"iteration"<<","
			<<"omega"<<","
			<<"theta"<<","
			<<"xi"<<","
			<<"rho"<<","
			<<"p"<<","
			<<"likelihood"<<"\n";
	}

		paramter_file
			<<simulation_counter<<","
			<<call_counter<<","
			<<omega<<","
			<<theta<<","
			<<xi<<","
			<<rho<<","
			<<p<<","
			<<mle<<"\n";

	//Printing out a status message to see whats going on
	if(call_counter++ % 100 == 0)
		cout<<" simulation counter = "<<simulation_counter
			<<" call_counter = "<<call_counter 
			<<" omega = "<<omega
			<<" theta = "<<theta
			<<" xi = "<<xi
			<<" rho = "<<rho
			<<" p = "<<p
			<<" sum = "<<mle
			<<endl;

}

VolParams get_next_VolParam_to_preturb(VolParams last_param, bool& result_improved, VolModel& current_model)
{
	if(result_improved)
		return last_param;
	switch(last_param)
	{//OMEGA, THETA, XI, ROE, P
	case OMEGA: return THETA;
	case THETA: return XI;
	case XI: return ROE;
	case ROE: return ((current_model != VAR_P) ? OMEGA : P);
	case P: return OMEGA;
	default: return OMEGA;
	}
}

void init_log(const VolModel model, const RunMode runmode, ofstream& log_stream)
{
	string logfile = "./output/log_" + get_model_name(model) + "_" +get_filter_type(filtertype)+ "_" + get_run_mode(runmode);
	cout<<"log file is: "<<logfile<<endl;
	log_stream.open(logfile.c_str());
	log_stream.setf(ios::fixed);
	log_stream<<setprecision(10);
}

string get_run_mode(const RunMode runmode) 
{
	switch(runmode)
	{
	case SIMPLE: return "simple";
	case NORMAL_RESIDUALS: return "normal_residuals";
	case UNCORRELATED_RESIDUALS: return "uncorrelated_normals";
	default: return "normal_and_uncorrelated_residuals";
	}
}

string get_model_name(const VolModel model) 
{
	switch(model)
	{
	case HESTON: return "heston";
	case GARCH: return "garch";
	case THREE_TWO: return "three_two";
	default: return "var_p";
	}
}

string get_filter_type(const FilterType filtertype)
{
	switch(filtertype)
	{
	case EKF: return "ekf"; 
	case UKF: return "ukf";
	}
}

void mark_better_parameters(int& simulation_counter,
	int& max_simulations,
	vol_params& current_best_params,
	vol_params& new_best_params,
	ofstream& log_stream,
	int price_count)
{
	log_stream<<"mark_better_parameters: We have found better parameters on simulation # "<<simulation_counter 
		<< " out of "<< max_simulations <<" on model "<<get_model_name(model)<<endl
		<<"mark_better_parameters: "<<" new mle                    = "<<new_best_params.get_mle()          <<" old = "<<current_best_params.get_mle()<<endl
		<<"mark_better_parameters: "<<" new traditional chi2       = "<<new_best_params.get_chi2()         <<" old = "<<current_best_params.get_chi2()<<endl
		<<"mark_better_parameters: "<<" new kalman chi2            = "<<new_best_params.get_kalman_chi2()  <<" old = "<<current_best_params.get_kalman_chi2()<<endl
		<<"mark_better_parameters: "<<" new mean adj. kalman chi2  = "<<new_best_params.get_adjusted_kalman_chi2()  <<" old = "<<current_best_params.get_adjusted_kalman_chi2()<<endl
		<<"mark_better_parameters: "<<" new omega                  = "<<new_best_params.get_omega()        <<" old = "<<current_best_params.get_omega()<<endl
		<<"mark_better_parameters: "<<" new theta                  = "<<new_best_params.get_theta()        <<" old = "<<current_best_params.get_theta()<<endl
		<<"mark_better_parameters: "<<" new xi                     = "<<new_best_params.get_xi()           <<" old = "<<current_best_params.get_xi()<<endl
		<<"mark_better_parameters: "<<" new rho                    = "<<new_best_params.get_rho()          <<" old = "<<current_best_params.get_rho()<<endl;
	if(model==VAR_P)
		log_stream<<"mark_better_parameters: "<<" new p                 = "<<new_best_params.get_p()   <<" old = "<<current_best_params.get_p()<<endl;

	log_stream<<endl<<endl;
	current_best_params.set_best_estimate(new_best_params, u, v, estimates, price_count);
}

string get_VolParams_string(VolParams& vp)
{
	switch(vp)
	{
	case OMEGA: return "OMEGA";
	case THETA: return "THETA";
	case XI: return "XI";
	case ROE: return "ROE";
	default: return "P";
	}
}

/*

//The function to minimize for the heston case
DP minimize_target_ekf_heston(Vec_I_DP & input);
//The function to minimize for the GARCH case
DP minimize_target_ekf_garch(Vec_I_DP & input);
//The function to minimize for the GARCH case
DP minimize_target_ekf_3_2(Vec_I_DP & input);
//The function to minimize for the variable p case
DP minimize_target_ekf_variable_p(Vec_I_DP & input);


//The function to minimize for the heston case
DP minimize_target_ekf_heston(Vec_I_DP & input) {
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];

	estimate_ekf_parm_1_dim(log_stock_prices, 
		muS, 
		n_stock_prices, 
		omega, 
		theta, 
		xi, 
		rho, 
		p_heston, //p_heston = 0.5
		u, 
		v, 
		estimates);

	double sum = 0;
	for(int i1 = 0; i1 < n_stock_prices; i1++)
		sum+=(log(v[i1])+u[i1]*u[i1]/v[i1]);

	//test for nan
	if(sum!=sum)
		return pow(10.00, 10.00);

	log_parameters(omega, theta, xi, rho, sum);

	return sum;
}

//The function to minimize for the GARCH case
DP minimize_target_ekf_garch(Vec_I_DP & input) {
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];

	estimate_ekf_parm_1_dim(log_stock_prices, 
		muS, 
		n_stock_prices, 
		omega, 
		theta, 
		xi, 
		rho, 
		p_garch, // p_garch = 1
		u, 
		v, 
		estimates);

	double sum = 0;
	for(int i1 = 0; i1 < n_stock_prices; i1++)
		sum+=(log(v[i1])+u[i1]*u[i1]/v[i1]);

	//test for nan
	if(sum!=sum)
		return pow(10.00, 10.00);

	log_parameters(omega, theta, xi, rho, sum);

	return sum;
}

//The function to minimize for the GARCH case
DP minimize_target_ekf_3_2(Vec_I_DP & input) {
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];

	estimate_ekf_parm_1_dim(log_stock_prices, 
		muS, 
		n_stock_prices, 
		omega, 
		theta, 
		xi, 
		rho, 
		p_3_2, // p_3_2 = 1.50
		u, 
		v, 
		estimates);

	double sum = 0;
	for(int i1 = 0; i1 < n_stock_prices; i1++)
		sum+=(log(v[i1])+u[i1]*u[i1]/v[i1]);

	//test for nan & rho going wild
	if(sum!=sum)
		sum = pow(10.00, 10.00);

	log_parameters(omega, theta, xi, rho, sum);

	return sum;
}

//The function to minimize for the GARCH case
DP minimize_target_ekf_variable_p(Vec_I_DP & input) {
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];
	double p = input[4];

	double sum = 0;

	//constrining rho 
	if(abs(rho) < 0.95) {
		estimate_ekf_parm_1_dim(log_stock_prices, 
			muS, 
			n_stock_prices, 
			omega, 
			theta, 
			xi, 
			rho, 
			p, //p is variable
			u, 
			v, 
			estimates);

		for(int i1 = 0; i1 < n_stock_prices; i1++)
			sum+=(log(v[i1])+u[i1]*u[i1]/v[i1]);
	} else {
		sum = pow(10.00, 6.00) * abs(rand()) * 1000.00; //if the optimizer tries unusual parameters, blow it up
	}

	//test for nan
	if(sum!=sum)
		return pow(10.00, 10.00);

	log_parameters(omega, theta, xi, rho, sum);

	return sum;
}
*/

/*
void ekf_filter(vector<double>& prices,
		string& parameter_file_name,
		string& residual_file_name)
{
    n_stock_prices = prices.size();
	u = new double[n_stock_prices];
	v = new double[n_stock_prices];
	estimates = new double[n_stock_prices + 1];
	//param variables are
	//self define class; defination in filter_utils.h
	 
	vol_params current_params(  //initial constructor 
	    0.02, //omega
		1.00,  //theta
		0.50, //xi
		-0.20, //rho
		1.00,  //p
		pow(10.00, 6), //mle
		pow(10.00, 6), //chi2
		pow(10.00, 6), //kalman_chi2
		pow(10.00, 6) //(mean) adjusted_kalman_chi2 
		); 
	vol_params best_params(current_params); //initial constructor
	
	Vec_IO_DP* start_; //files in nrtypes_lib.h self define type 
	
	Mat_IO_DP* identity_matrix; //files in nrtypes_lib.h self define type 

	if(model!=VAR_P) //if we aren't trying to estimate p, we only need a 4x4 matrix
	{
		//Initializing the starting point
		start_ = new Vec_IO_DP(4);
		identity_matrix = new Mat_IO_DP(4, 4);
	} else { //we are trying to estimate p, so we want a 5x5 matrix
		//Initializing the starting point
		start_ = new Vec_IO_DP(5);
		identity_matrix = new Mat_IO_DP(5, 5);
	}

	Vec_IO_DP& start = *start_; //aliasing the pointer for easy reference
	
	paramter_file.open(parameter_file_name.c_str());
	residual_file.open(residual_file_name.c_str());
	//setting the output to 10 decimal places
	paramter_file.setf(ios::fixed);
	paramter_file<<setprecision(10);
	residual_file.setf(ios::fixed);
	residual_file<<setprecision(10);

	double* classic_residuals = new double[n_stock_prices];
	double* kalman_residuals = new double[n_stock_prices];
	double* corrected_kalman_residuals = new double[n_stock_prices];

	double chi2; //buffer variable.
	
	while(simulation_counter <= max_simulations)
	{
		int nvars = (model!=VAR_P) ? 4 : 5; 
		for(int i = 0; i < nvars; i++) 
		{
			for(int j = 0; j < nvars; j++)
			{
				(*identity_matrix)[i][j] = (i==j) ? 1 : 0;
			}
		}

		current_params.populate_starting_vector(start);
		for(int i = 0; i < 5; i++)
			cout<<start[i]<<endl;
		int iter;
		DP mle = 0.00;
		
		NR::powell(start, *identity_matrix, ftol, iter, mle, minimize_target_ekf);
		
		if(model==VAR_P)
			cout<<" p = "<<current_params.get_p();
		cout<<endl<<endl;

		
		// Setting the current params out of the optimizations.
		
		current_params.extract_params_from_vector(start);
		current_params.set_mle(mle);
		
		//build_classic_xx are definded in filter_utils.cpp
		build_classic_residuals(prices, estimates, classic_residuals);
        build_kalman_residuals(prices, estimates, v, kalman_residuals);
        build_mean_corrected_kalman_residuals(prices, estimates, u, v, corrected_kalman_residuals);
		//getting the chi2 statistic for the classic residuals
		is_normal(classic_residuals, true, prices.size(), chi2); current_params.set_chi2(chi2);
		is_normal(kalman_residuals, true, prices.size(), chi2); current_params.set_kalman_chi2(chi2);
		is_normal(corrected_kalman_residuals, true, prices.size(), chi2); current_params.set_adjusted_kalman_chi2(chi2);

		if(runmode == NORMAL_RESIDUALS) 
		// my code, log_file
		log_better_param(log_file,
			best_params, 
			current_params, 
			prices);
						
		simulation_counter++;
		paramter_file.flush(); //flush the buffer to disc		
	}
	
	log_residual(residual_file, 
		best_params, 
		current_params,
		prices, 
		classic_residuals,  
		kalman_residuals,  
		corrected_kalman_residuals);

    //the ending of the log_file
	log_stats(log_file,
		classic_residuals,
		kalman_residuals,
		corrected_kalman_residuals,
		prices.size());

	residual_file.close();
	paramter_file.close();

	delete[] u, v, estimates, classic_residuals, kalman_residuals, corrected_kalman_residuals;
	delete start_, identity_matrix;
}*/

/*
DP minimize_target_ekf(Vec_I_DP & input) 
{
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];
	double p;
	switch(model) 
	{
	case HESTON: p = p_heston; break;
	case GARCH: p = p_garch; break;
	case THREE_TWO: p = p_3_2; break;
	default: p = input[4];
	}

	estimate_ekf_parm_1_dim(log_stock_prices, 
		muS, 
		n_stock_prices, 
		omega, 
		theta, 
		xi, 
		rho, 
		p, //p as specified by the model
		u, 
		v, 
		estimates);
		
	double mle = 0.00;

	for(int i1 = 0; i1 < n_stock_prices; i1++)
		mle+=(log(v[i1])+u[i1]*u[i1]/v[i1]);

	//test for nan
	if(mle!=mle || 
		rho > 0.95 || rho < -0.95
		|| omega < 0
		|| xi < 0
		|| theta < 0
		|| p < 0.25 || p > 2)
		mle = pow(10.00, 5.00) + rand(); //add a random number to it so it doesn't settle there

	log_parameters(omega, theta, xi, rho, p, mle);

	return mle;
}*/
