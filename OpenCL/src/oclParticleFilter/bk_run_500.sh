./bin/final_ekf ./io/Final-Project.csv ./io/var_p_param.csv ./io/var_p_residuals.csv 4 2 500 &
./bin/final_ekf ./io/Final-Project.csv ./io/heston_parms.csv ./io/heston_residuals.csv 1 2 500 &
./bin/final_ekf ./io/Final-Project.csv ./io/garch_params.csv ./io/garch_residuals.csv 2 2 500 &
./bin/final_ekf ./io/Final-Project.csv ./io/three_two_params.csv ./io/three_two_residuals.csv 3 2 500 &



# test only
./final_ekf ../../io/Final-Project.csv ../../io/three_two_params.csv ./io/three_two_residuals.csv 3 2 5 &

