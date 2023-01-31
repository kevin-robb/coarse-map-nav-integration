#include "localization_pkg/ekf.hpp"

// init the EKF.
EKF::EKF() {
    // set the noise covariance matrices.
    this->V.setIdentity(2,2);
    this->W.setIdentity(2,2);
    // initialize state distribution.
    this->x_t.resize(3);
    this->x_t.setZero(3); // set starting vehicle pose.
    this->x_pred.setZero(3);
    this->P_t.setIdentity(3,3);
    this->P_t(0,0) = 0.01 * 0.01;
    this->P_t(1,1) = 0.01 * 0.01;
    this->P_t(2,2) = 0.005 * 0.005;
    this->P_pred.setIdentity(3,3);
    this->P_pred(0,0) = 0.01 * 0.01;
    this->P_pred(1,1) = 0.01 * 0.01;
    this->P_pred(2,2) = 0.005 * 0.005;
    // set jacobians that are constant.
    this->H_w.setIdentity(2,2);
}