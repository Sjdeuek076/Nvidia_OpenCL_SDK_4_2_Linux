##program, input_file, output_param, output_residuals, model, runmode, filter type, simulation times.
./bin/final_ekf input/Final-Project.csv ./output/var_p_param_ekf.csv ./output/var_p_residuals_ekf.csv 4 2 1 2 &
./bin/final_ekf input/Final-Project.csv ./output/heston_parms_ekf.csv ./output/heston_residuals_ekf.csv 1 2 1 2 &
./bin/final_ekf input/Final-Project.csv ./output/garch_params_ekf.csv ./output/garch_residuals_ekf.csv 2 2 1 2 &
./bin/final_ekf input/Final-Project.csv ./output/three_two_params_ekf.csv ./output/three_two_residuals_ekf.csv 3 2 1 2 &


./bin/final_ekf input/Final-Project.csv ./output/var_p_param_ukf.csv ./output/var_p_residuals_ukf.csv 4 2 2 2 &
./bin/final_ekf input/Final-Project.csv ./output/heston_parms_ukf.csv ./output/heston_residuals_ukf.csv 1 2 2 2 &
./bin/final_ekf input/Final-Project.csv ./output/garch_params_ukf.csv ./output/garch_residuals_ukf.csv 2 2 2 2 &
./bin/final_ekf input/Final-Project.csv ./output/three_two_params_ukf.csv ./output/three_two_residuals_ukf.csv 3 2 2 2 &





./final_ekf ../../input/Final-Project.csv ../../output/var_p_param_ukf.csv ../../output/var_p_residuals_ukf.csv 4 2 2 2 &
