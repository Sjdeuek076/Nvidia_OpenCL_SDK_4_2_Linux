#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <string>
#include "./recipes/nr.h"
#include "filters.h"
#include <vector>
#include "filter_utils.h"
#include "pstream.h" 

using namespace std;
using namespace hogehoge;




enum VolModel {HESTON = 1, GARCH = 2, THREE_TWO = 3, VAR_P = 4};
enum RunMode {SIMPLE = 1, NORMAL_RESIDUALS = 2, UNCORRELATED_RESIDUALS = 3, NORMAL_AND_UNCORRELATED_RESIDUALS = 4};
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

int call_counter = 0;
int simulation_counter = 1;

ofstream paramter_file;  
ofstream residual_file;  
ofstream log_file;

const double ftol = 1.00e-6;

void parse_args(int argc, char** argv, 
	string& input_file_name,
	string& parameter_file_name,
	string& residual_file_name);
	
//ivf p.58 try to minimize sum(c_{model}(K_j) - c_{mkt}(K_j))^2
//where c stand for corresponding option prices
//and k for strike prices
//minimization can be done via the direction set (Powell) method.
DP minimize_target_unscented_kalman_parameters_1_dim(Vec_I_DP & input);
//void gplot();

int main(int argc, char** argv) {
	
	srand(time(NULL)); //seed the random number generator

	vector<double> prices;
	string input_file_name, parameter_file_name = "", residual_file_name = "";
	
	// parse args <input><output_parameter><output_residual><model 1:4><runmode=1><max-simulation>
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
	u = new double[n_stock_prices];
	v = new double[n_stock_prices];
	estimates = new double[n_stock_prices + 1];

    //log stock prices
	for(unsigned int i = 0; i < prices.size(); i++) {
		log_stock_prices[i] = log(prices[i]);
	}

	//param variables are
	vol_params current_params(0.02, //omega
		1.00,  //theta
		0.50, //xi
		-0.20, //rho
		1.00,  //p
		pow(10.00, 6), //mle
		pow(10.00, 6), //chi2
		pow(10.00, 6), //kalman_chi2
		pow(10.00, 6) //(mean) adjusted_kalman_chi2 
		); 
	vol_params best_params(current_params);
	
	Vec_IO_DP* start_;
	
	Mat_IO_DP* identity_matrix;

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

	double* classic_residuals = new double[prices.size()];
	double* kalman_residuals = new double[prices.size()];
	double* corrected_kalman_residuals = new double[prices.size()];

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
		
		NR::powell(start, *identity_matrix, ftol, iter, mle, minimize_target_unscented_kalman_parameters_1_dim);
		
		if(model==VAR_P)
			cout<<" p = "<<current_params.get_p();
		cout<<endl<<endl;

		/*
		* Setting the current params out of the optimizations.
		*/
		current_params.extract_params_from_vector(start);
		current_params.set_mle(mle);

		build_classic_residuals(prices, estimates, classic_residuals);
        build_kalman_residuals(prices, estimates, v, kalman_residuals);
        build_mean_corrected_kalman_residuals(prices, estimates, u, v, corrected_kalman_residuals);
		//getting the chi2 statistic for the classic residuals
		is_normal(classic_residuals, true, prices.size(), chi2); current_params.set_chi2(chi2);
		is_normal(kalman_residuals, true, prices.size(), chi2); current_params.set_kalman_chi2(chi2);
		is_normal(corrected_kalman_residuals, true, prices.size(), chi2); current_params.set_adjusted_kalman_chi2(chi2);

		if(runmode == NORMAL_RESIDUALS) 
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
			current_params.set_roe(((rand()/rand_max) * 2.00) - 1.00);
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
			log_file<<"old rho   "<<best_params.get_roe()  
				<<" new-> "<<current_params.get_roe()
				<<" diff= "<<best_params.get_roe() - current_params.get_roe()<<endl;
			if(model==VAR_P)
				log_file<<"old p     "<<best_params.get_p()
					<<" new-> "<<current_params.get_p()<<" diff= "
					<<best_params.get_p() - current_params.get_p()<<endl;

			log_file<<endl<<endl<<endl;


			call_counter = 0;
			
		} 
		
		simulation_counter++;
		paramter_file.flush(); //flush the buffer to disc
	}

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
			<<"--roe   "<<current_params.get_roe()<<endl
			<<"--xi    "<<current_params.get_xi()<<endl;
		if(model==VAR_P)
			log_file<<"--p     "<<current_params.get_p()<<endl;
		for(unsigned int i = 0; i < prices.size(); i++) {
			residual_file<<i<<"."<<prices[i]<<","<<exp(estimates[i])<<","
				<<classic_residuals[i]<<","
				<<kalman_residuals[i]<<","
				<<corrected_kalman_residuals[i]<<","
				<<endl;
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
			<<"--roe   "<<best_params.get_roe()<<endl
			<<"--xi    "<<best_params.get_xi()<<endl;
		if(model==VAR_P)
			log_file<<"--p     "<<best_params.get_p()<<endl;
		for(unsigned int i = 0; i < prices.size(); i++) {
			residual_file<<i<<","<<prices[i]<<","
				<<exp(estimates[i])<<","
				<<classic_residuals[i]<<","
                                <<kalman_residuals[i]<<","
                                <<corrected_kalman_residuals[i]<<","
                                <<endl;
		}
	}

	log_stats(log_file,
		classic_residuals,
		kalman_residuals,
		corrected_kalman_residuals,
		prices.size());

	residual_file.close();
	paramter_file.close();

	delete[] log_stock_prices, u, v, estimates, classic_residuals, kalman_residuals, corrected_kalman_residuals;
	delete start_, identity_matrix;
	/*
	cout<<"Press any key to continue"<<endl;
	cin.get();
	*/
	return 0;
}

/*void gplot()
{
	const char *filename= "filter_estimate.data";
	ps::pipestream gnuplot("gnuplot -presistence");
	gnuplot<<"set grid"<<ps::endl;
	gnuplot<<"set nokey"<<ps::endl;
	gnuplot<<"set xlabel 'X'"<<ps::endl;
	gnuplot<<"set ylabel 'Y'"<<ps::endl;
	//gnuplot<<"set xrange[1:220]"<<ps::endl;
	ofstream ofs(filename);
	for(int h = 0; h <= n_stock_prices-1; h++){
		//ofs<<h<<'\t'<<u<<'\t'<<v<<'t'<<estimates[h]<<endl;
		ofs<<h<<'\t'<<estimates[h]<<endl;
		gnuplot<<"plot '"<<filename<<"' using 1:2 w l lt 3"<<ps::endl;
		gnuplot<<"reread"<<ps::endl;   
		system("sleep 0.001");
	}
	ofs.close();
}*/

void parse_args(int argc, char** argv,
	string& input_file_name,
	string& parameter_file_name,
	string& residual_file_name) {

	string usage = "syntax is program_name <input_file> <parameter_output_file> <residual_output_file> "
		"<MODEL 1=HESTON, 2=GARCH, 3=3_2, 4=variable p> "
		"<RUNMODE 1=SIMPLE, 2=NORMAL_RESIDUALS>. "
		"<MAXSIMULATIONS: Required if RUNMODE!=1> \n "
		"The input file should be a 1 columned csv price file, the output file(s) will be created";

	//Parse the command line args.
	try {
		if(argc < 6) 
			throw usage;
		input_file_name = string(argv[1]);
		
		parameter_file_name = string(argv[2]);
		residual_file_name = string(argv[3]);
		
		string model_ = string(argv[4]);
		string runmode_ = string(argv[5]);
		if(argc==7) 
		{
			max_simulations = atoi(argv[6]);
			cout<<"max_simulations is set to "<<max_simulations<<endl;
		}

		std::cout<<"input file is "<<input_file_name<<endl;
		std::cout<<"parameter_output_file is "<<parameter_file_name<<endl;
		std::cout<<"residual_output_file is "<<residual_file_name<<endl;

		if(model_!="1" &&
			model_!="2" &&
			model_!="3" &&
			model_!="4")
			throw "model is not one of 1,2,3,4. Usage is \n" + usage;

		if(runmode_!="1" &&
			runmode_!="2")
			throw "runmode is not one of 1,2. Usage is \n" + usage;

		if(runmode_!="1" && argc!=7)
			throw "RUNMODE!=1 and MAXSIMULATIONS not provided";

		//convert them into int's
		model = (VolModel) atoi(model_.c_str()); 
		runmode = (RunMode) atoi(runmode_.c_str());
		
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

DP minimize_target_unscented_kalman_parameters_1_dim(Vec_I_DP & input) {
	double omega = input[0];
	double theta = input[1];
	double xi = input[2];
	double rho = input[3];

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

	double sum = 0;
	if(abs(rho) > 1.00) {
		sum = 1000.00;
	} else {
		for(int i1 = 0; i1 < n_stock_prices; i1++)
			sum+=(log(v[i1])+u[i1]*u[i1]/v[i1]);
	}

	//print the header first time around
	/*if(call_counter==0 && !output_file_name.empty()) {
		output_file<<"iteration"<<","
			<<"omega"<<","
			<<"theta"<<","
			<<"xi"<<","
			<<"rho"<<","
			<<"likelihood"<<endl;
	}*/

	output_file<<call_counter<<'\t'
			<<omega<<'\t'
			<<theta<<'\t'
			<<xi<<'\t'
			<<rho<<'\t'
            <<sum<<endl;

//gunplot code
			gnuplot<<"plot '"
			<<output_file_name<<"' using 1:2 title 'omega' w l lt 1,\\"<<ps::endl<<"'"
			<<output_file_name<<"' using 1:3 title 'theta' w l lt 2,\\"<<ps::endl<<"'"
			<<output_file_name<<"' using 1:4 title 'xi' w l lt 3,\\"<<ps::endl<<"'"
			<<output_file_name<<"' using 1:5 title 'rho' w l lt 4"<<ps::endl;
		gnuplot<<"reread"<<ps::endl;
		//system("sleep 0.001");
	//}


	//Printing out a status message to see whats going on
	if(call_counter++ % 100 == 0){
		cout<<" call_counter = "<<call_counter 
			<<" omega = "<<omega
			<<" theta = "<<theta
			<<" xi = "<<xi
			<<" rho = "<<rho
			<<" sum = "<<sum
			//<<" *u = "<<u[call_counter]
			//<<" *v = "<<v[call_counter] 
			//<<" *estimates = "<<estimates[call_counter]
			<<endl;
	 /*for(int h = 0; h <= n_stock_prices-1; h++)
		output_file<<estimates[h]<<",";
		output_file<<endl;
	*/
		}


	return sum;
}
