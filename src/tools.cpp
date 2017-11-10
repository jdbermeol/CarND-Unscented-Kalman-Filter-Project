#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  if(estimations.size() != ground_truth.size()
          || estimations.size() == 0){
    cout << "Invalid estimation or ground_truth data" << endl;
    VectorXd rmse(1);
    rmse << -1;
    return rmse;
  }

  VectorXd rmse = estimations[0];
  rmse.setZero();
 
  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //return the result
  return rmse.array().sqrt();
}
