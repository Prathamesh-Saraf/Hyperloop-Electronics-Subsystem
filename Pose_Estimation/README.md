# Welcome to the Pose estimation repository.

A 1-Dimensional modified Kalman Filter is currently being used. 

# Acceleration
For the acceleration model, the prediction step is carried out by taking the value from the acceleration-time graph at discrete time intervals. For the update step, the acceleration is measured from the IMU. The output after the update step returns a mean updated value of the acceleration along with the updated covariance. 

For acceleration, the ideal or ground-truth curve was approximated by scaling down a base-10 logarithmic curve followed by scaled step-functions for the negative acceleration period. As is visible, the peak occurs at the beginning at 260m/s2. The measurement noise was assumed to be gaussian in nature and was sampled 120 times with a scaled standard deviation of 10. Similarly, the process noise was defined, however, with a standard deviation of 5. A one-dimensional Kalman filter was then used. It was seen that the RMSE for raw sensor output was 53.84 whereas the RMSE for Kalfor the velocity prediction step. The process and measurement noise were again defined as gaussian with standard deviations of 35 and 10. It was seen that the RMSE of the raw sensor output was 35.97 compared to an RMSE of 11.73 with the Kalman filter. man filtering was just 4.44.

# Velocity
The velocity model does not require simulated data, however, considers the acceleration estimate obtained in the previous step. The prediction step is carried out through simple equations of motion and involves the acceleration estimate output. For the update step, the sensor model includes the tachometer output multiplied with the transformation matrix that provides measurement of the velocity. The velocity estimate output is computed in the form a mean and covariance matrix for a normal distribution.

For the velocity prediction step, the process and measurement noise were again defined as gaussian with standard deviations of 35 and 10. It was seen that the RMSE of the raw sensor output was 35.97 compared to an RMSE of 11.73 with the Kalman filter. 

# Position
Finally, the position model is calculated that considers displacement from the start of the track.  For the prediction step, the position is calculated using a simple kinematic equation of motion taking the velocity estimate and acceleration estimate. The update step involves the pre-multiplication with the transformation matrix with the optical encoder output. The output received is the position from start along the direction of motion and is expressed as a normal distribution. 

For the position parameter, a similar Kalman filter was used, however the RMSE was higher and comparatively inconsistent as the acceleration and velocity estimates. This however, can be attributed to the fact that the scales for each parameter vary in terms of the power of 10. Hence, relatively speaking, the Kalman filter was successful in keeping the percentage RMSE very low. Moreover, in the real world, re-calibration of the sensors and Kalman gains happens with crossing of every fiducial marker as well as end points. Hence, this would essentially reduce the error further.
