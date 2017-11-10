#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_/ (lambda_ + n_aug_);
  for (int i=1; i< 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  R_readar_ = MatrixXd(3, 3);
  R_readar_ << std_radr_*std_radr_, 0, 0,
               0, std_radphi_*std_radphi_, 0,
               0, 0,std_radrd_*std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0,std_laspy_*std_laspy_;
}

UKF::~UKF() {}

double UKF::NormAng(double a) {
  //angle normalization
  while (a > M_PI) a -= 2. * M_PI;
  while (a < -M_PI) a += 2. * M_PI;
  return a;
}

MatrixXd UKF::GenerateSigmaPoints() {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i < n_aug_; i++){
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  return Xsig_aug;
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug, double delta_t) {
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  //predict sigma points
  for (int i = 0; i< 2 * n_aug_ + 1; i++){
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }
    
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;    

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p += nu_a*delta_t;    

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;    
  }
  return Xsig_pred;
}

VectorXd UKF::Mean(const MatrixXd &X) {
  return X * weights_; 
}

MatrixXd UKF::Varianze(const MatrixXd &A, const MatrixXd &B){
  MatrixXd P = weights_(0) * A.col(0) * B.col(0).transpose();
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    P = P + weights_(i) * A.col(i) * B.col(i).transpose();
  }
  return P;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = GenerateSigmaPoints();
  Xsig_pred_ = PredictSigmaPoints(Xsig_aug, delta_t);
  x_ = Mean(Xsig_pred_);
  X_diff_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    X_diff_.col(i) = Xsig_pred_.col(i) - x_;
    X_diff_(3, i) = NormAng(X_diff_(3, i));
  }
  P_ = Varianze(X_diff_, X_diff_);
}

double UKF::UpdateUKF(const MeasurementPackage &meas_package, MatrixXd (UKF::*measure)(), VectorXd (UKF::*fix_vector)(const VectorXd&), const MatrixXd &R) {
  MatrixXd measures = (this->*measure)();
  VectorXd z_pred = Mean(measures);

  MatrixXd Z_diff = MatrixXd(z_pred.size(), 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    Z_diff.col(i) = (this->*fix_vector)(measures.col(i) - z_pred);
  }

  MatrixXd S =  Varianze(Z_diff, Z_diff) + R;
  MatrixXd T = Varianze(X_diff_, Z_diff);
  MatrixXd Si = S.inverse();
  MatrixXd K = T * Si;
  VectorXd y = (this->*fix_vector)(meas_package.raw_measurements_ - z_pred);
  x_ = x_ + K * y;
  P_ = P_ - K * S * K.transpose();
  return y.transpose() * Si * y;
}

MatrixXd UKF::MeasurePredictedSigmaPointsRadar() {
  MatrixXd Z = MatrixXd(3, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Z(0,i) = sqrt(p_x*p_x + p_y*p_y);
    Z(1,i) = atan2(p_y,p_x);
    Z(2, i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);
  }
  return Z;
}

VectorXd UKF::FixRadarVector(const VectorXd &z) {
  VectorXd y = z;
  y(1) = NormAng(y(1));
  return y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
double UKF::UpdateRadar(const MeasurementPackage &meas_package) {
  return UpdateUKF(meas_package, &UKF::MeasurePredictedSigmaPointsRadar,  &UKF::FixRadarVector, R_readar_);
}

MatrixXd UKF::MeasurePredictedSigmaPointsLidar() {
  MatrixXd Z = Xsig_pred_.topRows(2);
  return Z;
}

VectorXd UKF::FixLidarVector(const VectorXd &z) {
  return z;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
double UKF::UpdateLidar(const MeasurementPackage &meas_package) {
  return UpdateUKF(meas_package, &UKF::MeasurePredictedSigmaPointsLidar,  &UKF::FixLidarVector, R_lidar_);
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
double UKF::ProcessMeasurement(const MeasurementPackage &meas_package) {

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) {
    return -1;
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_) {
    return -1;
  }

  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;

    x_ = VectorXd(n_x_);
    P_ = MatrixXd(n_x_, n_x_);
    P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 10, 0, 0,
            0, 0, 0, 10, 0,
            0, 0, 0, 0, 10;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
       float rho = meas_package.raw_measurements_[0];
       float phi = meas_package.raw_measurements_[1];
       float rho_dot = meas_package.raw_measurements_[2];
       float px = rho * cos(phi);
       float py = rho * sin(phi);
       float vx = rho_dot * cos(phi);
       float vy = rho_dot * sin(phi);
       float v  = sqrt(vx * vx + vy * vy);
       x_ << px, py, v, 0, 0;
       P_(2, 2) = 1;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    is_initialized_ = true;
    return -1;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    return UpdateRadar(meas_package);
  } else {
    return UpdateLidar(meas_package);
  }
}
