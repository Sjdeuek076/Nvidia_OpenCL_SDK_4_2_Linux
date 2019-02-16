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

using namespace std;

int main() {

	vector<double> file_data;
	string file_name = "C:/users/hildingur/nvs.csv";
	read_lines(file_name, file_data);

	int count = file_data.size();
	double* data = new double[count];
	for(int i = 0; i < count; i++)
		data[i] = file_data[i];

	//bool is_normal(double* data, int data_count, double confidence_interval = 0.95);
	if(is_normal(data, count))
		cout<<"Data is normal"<<endl;
	else
		cout<<"Data is NOT normal"<<endl;

	cin.get();

}