/*
 * File:   bpnn.h
 * Author: emmanuj
 *
 * Created on November 29, 2015, 12:35 AM
 */

#ifndef BPNN_H
#define BPNN_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <iomanip>      // std::setprecision

class BPNN {
private:
    float learnRate = 0.01;
    const float momentum = 0.01;
    int input_count; // # number of input nodes
    int out_count; // # number of output nodes
    int hidden_count; // # hidden nodes
    std::vector< std::vector<double> > in_weights;
    std::vector< std::vector<double> > out_weights;
    std::vector< std::vector<double> > cur_in_weights;
    std::vector< std::vector<double> > cur_out_weights;
    std::vector<double> input_xi;
    std::vector<double> hidden_out;
    std::vector<double> final_out;

public:

    BPNN(int n_in, int n_hidden, int n_out) :
    input_count(n_in+1),
    out_count(n_out),
    hidden_count(n_hidden),
    in_weights(),
    out_weights(),
    cur_in_weights(),
    cur_out_weights(),
    input_xi(n_in + 1, 1.0), // plus one bias node
    hidden_out(n_hidden, 1.0),
    final_out(n_out, 1.0)
    {
        generateRandomMatrix(in_weights, input_count, hidden_count, -0.2, 0.2);
		generateRandomMatrix(out_weights, hidden_count, out_count, -1, 1);
		initializeMatrix(cur_in_weights, input_count, hidden_count);
		initializeMatrix(cur_out_weights, hidden_count, out_count);
    }

    void initializeMatrix(std::vector< std::vector<double> >& mat, int nrow, int ncol) {
        mat = std::vector< std::vector<double> >(nrow);
        for (int i = 0; i < nrow; i++) {
            mat[i] = std::vector<double>(ncol);
        }
    }

    void generateRandomMatrix(std::vector< std::vector<double> >& mat, int nrow, int ncol, double lb, double ub) {
		mat = std::vector< std::vector<double> >(nrow);
		for (int i = 0; i < nrow; i++) {
			mat[i] = std::vector<double>(ncol);
            for (int j = 0; j < ncol; j++) {
                mat[i][j] = genRandom(lb, ub);
            }
        }
    }

    double genRandom(float hi, float lo) {
        return lo + (static_cast<float> (rand()) / static_cast<float> (RAND_MAX / (hi - lo)));
    }

	//void normalize

	void feed_forward(std::vector<double>& inputs){

		if (inputs.size() != input_count - 1){
			std::cout<<"Wrong number of inputs"<<std::endl;
			exit(1);
		}

		//copy inputs over. exclude the last index
		input_xi = std::vector<double>(inputs.begin(), inputs.end() - 1);

		// input from in nodes to hidden nodes is weight and activated
		for(int j=0; j<hidden_count;j++ ){
			double sum = 0.0;
			for (int i = 0; i < input_count; i++) {
				sum = sum + input_xi[i] * in_weights[i][j];
			}
			hidden_out[j] = tanh(sum);
		}

		// input from hidden nodes to output nodes is further weight and activated
		for(int k=0; k<out_count;k++ ){
			double sum = 0.0;
			for (int j = 0; j < hidden_count; j++) {
				sum = sum + hidden_out[j] * out_weights[j][k];
			}
			final_out[k] = tanh(sum);
		}
	}

	double back_propagate(std::vector<double>& targets){

		if (targets.size() != out_count){
			std::cout<<"Wrong number of outputs"<<std::endl;
			exit(1);
		}

		//error terms for output
		std::vector<double> output_deltas(out_count);
		for(int k=0;k<out_count;k++){
			const double error = targets[k] - final_out[k];
			output_deltas[k] = ( 1.0 - pow(final_out[k],2)) * error;
		}

		//  calculate error terms for hidden
		std::vector<double> hidden_deltas(hidden_count);
		for(int j=0;j< hidden_count;j++){
			double error = 0.0;
			for(int k=0;k<out_count;k++){
				error = error + output_deltas[k] * out_weights[j][k];
			}
			hidden_deltas[j] = (1.0 - pow(hidden_out[j],2)) * error;
		}

		// update output weights
		for(int j=0;j<hidden_count;j++){
			for( int k=0; k< out_count;k++){
				const double w_change = output_deltas[k]* hidden_out[j];
				out_weights[j][k] = out_weights[j][k] + learnRate * w_change + momentum * cur_out_weights[j][k];
				cur_out_weights[j][k] = w_change;
			}
		}
		// update input weights
		for(int i=0;i<input_count;i++){
			for(int j=0;j<hidden_count;j++){
				const double w_change = hidden_deltas[j] * input_xi[i];
				in_weights[i][j] = in_weights[i][j] + learnRate * w_change + momentum * cur_in_weights[i][j];
				cur_in_weights[i][j] = w_change;
			}
		}
		// calculate error
		double error = 0.0;
		for(int k=0;k<targets.size();k++){
			error = error + 0.5 * pow((targets[k] - final_out[k]),2);
		}
		return error;
	}

	void train(std::vector< std::vector< std::vector<double> > >& tdata, int niter){
		std::streamsize ss = std::cout.precision();
		for(int i=0;i< niter;i++){
			double error = 0.0;
			for(int j=0;j< tdata.size();j++){
				feed_forward(tdata[j][0]);
				error += back_propagate(tdata[j][1]);
			}
			if(i% 10 == 0){
				std::cout<<"Error "<<std::setprecision(5)<<error<<ss<<std::endl;
			}
			if(error < 4){
				std::cout<<"Error "<<std::setprecision(5)<<error<<ss<<std::endl;
				break;
			}
			//std::cout<<i<<std::endl;
		}
	}

	void test(std::vector< std::vector<double> >& tdata){
		for(int i=0;i< tdata.size();i++){
			feed_forward(tdata[i]);
			std::cout<<"[";
			for(int j=0;j<tdata[i].size();j++){
				if(j != tdata[i].size() -1){
					std::cout<<tdata[i][j]<<",";
				}else{
					std::cout<<tdata[i][j]<<"]";
				}
			}
			std::cout<<" ==> [";
			for(int j=0;j<final_out.size();j++){
				if(j != final_out.size() -1){
					std::cout<<final_out[j]<<",";
				}else{
					std::cout<<final_out[j]<<"]";
				}
			}
			std::cout<<std::endl;
		}
	}

	void printWeights(){
		std::cout<<" ===== input weights ===== "<<std::endl;
		for(std::vector<double>& v:in_weights){
			for(double& a:v){
				std::cout<<a<<" ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;

		std::cout<<" ===== output weights ===== "<<std::endl;
		for(std::vector<double>& v:out_weights){
			for(double& a:v){
				std::cout<<a<<" ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}
};

#endif /* BPNN_H */
