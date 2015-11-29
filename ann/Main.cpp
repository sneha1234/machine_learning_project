/*
 * File:   main.cpp
 * Author: Emmanuel John
 *
 * Created on November 29, 2015, 12:34 AM
 */

#include <cstdlib>
#include <ctime>
#include "bpnn.h"
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include "util.h"

int main(int argc, char** argv) {
    srand (static_cast <unsigned> (time(0)));

	//params
	int n_in = 5;
	int n_out = 39;
	int n_hidden = 5;

	// read data
	std::string tfilename(argv[1]);
	std::ifstream train_file(tfilename);
	std::string line;
	char delim = ',';
	std::vector< std::vector< std::vector<double> > > train_data;
	while(std::getline(train_file, line)){
		line = Util::trim(line);
		std::vector<double> values = Util::split_numeric(line, delim);


		std::vector<double> inp(values.begin(), values.end()-1);
		std::vector<double> cls(n_out);
		cls[values[values.size()-1]] = 1.0;

		std::vector< std::vector<double> > data(2);
		data[0] = inp;
		data[1] = cls;

		train_data.push_back(data);
	}

	BPNN bp(n_in,n_hidden,n_out);
	//bp.printWeights();
	bp.train(train_data, 10000);
	//bp.printWeights();

	//test
	double test_data[] = {1.43157558,3,0,-122.425891675,37.7745985957};
	std::vector<double> t (test_data, test_data + sizeof(test_data) / sizeof(double) );

	std::vector<std::vector<double> > v;
	v.push_back(t);

	bp.test(v);

    return 0;
}
