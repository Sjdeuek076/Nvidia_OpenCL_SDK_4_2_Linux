#/bin/sh

rm ./io/*residu*csv -v
rm ./io/**params*csv -v

./bin/final_ekf "./io/Final-Project.csv" "./io/heston_ekf_params.csv" "./io/heston_ekf_residuals.csv" 1

./bin/final_ekf "./io/Final-Project.csv" "./io/garch_ekf_params.csv" "./io/garch_ekf_residuals.csv" 2

./bin/final_ekf "./io/Final-Project.csv" "./io/3-2_ekf_params.csv" "./io/3-2_ekf_residuals.csv" 3

./bin/final_ekf "./io/Final-Project.csv" "./io/free-p_ekf_params.csv" "./io/free-p_ekf_residuals.csv" 4
