import numpy as np
import matplotlib.pyplot as plt

##Input step function
#ground_truth = 10*np.ones(100)
#ground_truth[20:] = ground_truth[20:] - 10
#ground_truth[80:] = ground_truth[80:] - 10


######################Acceleration Start##########################
a = np.logspace(7,0,num=120)
a = 2.6*a/100000 
a[50:90] = a[50:90] - 35


ground_truth = a.copy()

noise_obs = np.random.normal(0,10,120)#Q
noise_model = np.random.normal(0,5,120)#R
simulated_accel = ground_truth + noise_model
observation = ground_truth + noise_obs
cov_sensor_model = np.cov(noise_obs) #Rt
cov_dynamic_model = np.cov(noise_model) #Qt 
#print(cov_sensor_model) ##Checking value of covaraince
#print(cov_dynamic_model) ##Checking value of covaraince
estimated_acceleration = []

##Initial belief params
accel_initial=ground_truth[0]
best_accel_estimate = 0
##Define model
i=0
while i<120:
    accel_initial= simulated_accel[i]
    kalman_gain = (cov_dynamic_model)/(cov_dynamic_model + cov_sensor_model)
    best_accel_estimate = accel_initial + kalman_gain*(observation[i]-simulated_accel[i])
    estimated_acceleration.append(best_accel_estimate)
    i = i+1

#plt.plot(range(len(ground_truth)), ground_truth,label = "Ground Truth")
#plt.plot(range(len(observation)), observation,label = "Sensor value")
#plt.plot(range(len(estimated_acceleration)),estimated_acceleration,label = "Kalman filter")
#plt.plot(range(len(estimated_acceleration)),noise_obs,label = "Noise")
#plt.legend()
#plt.show()
##################Acceleration Over##############################
##################Velocity start##############################
##Defining curves
##Ideal ground truth###
v = np.zeros(120)

for i in range(1,120):
    v[i] = v[i-1] + a[i]
    
ground_truth_velocity = v.copy()

##Get the predicted velocity values for 
v_predicted = np.zeros(120)
accel_estimate = np.asarray(estimated_acceleration)

for i in range(1,120):
    v_predicted[i] = v_predicted[i-1] + accel_estimate[i]

##Add noise to the velocity models
noise_obs_velocity = np.random.normal(0,35,120)#Q
noise_model_velocity = np.random.normal(0,10,120)#R
#simulated_vel = v_predicted + noise_model_velocity
simulated_vel = v_predicted
observation_vel = ground_truth_velocity + noise_obs_velocity
cov_sensor_model_vel = np.cov(noise_obs_velocity) #Rt
cov_dynamic_model_vel = np.cov(noise_model_velocity) #Qt 


##KF for Velocity
estimated_velocity = []

##Initial belief params
velocity_initial=ground_truth[0]
best_vel_estimate = 0

##Define model
i=0
while i<120:
    velocity_initial= simulated_vel[i]
    kalman_gain_vel = (cov_dynamic_model_vel)/(cov_dynamic_model_vel + cov_sensor_model_vel)
    best_vel_estimate = velocity_initial + kalman_gain_vel*(observation_vel[i]-simulated_vel[i])
    estimated_velocity.append(best_vel_estimate)
    i = i+1
 
kf_est_velocity = np.array(estimated_velocity)

#Plotting the results
#plt.plot(range(len(ground_truth_velocity)), ground_truth_velocity,label = "Ground Truth Velocity")
#plt.plot(range(len(observation_vel)), observation_vel,label = "Sensor value for Velocity")
#plt.plot(range(len(estimated_velocity)),kf_est_velocity,label = "Kalman filter Velocity")
#plt.plot(range(len(v_predicted)),v_predicted,label = "predicted velocity")
#plt.legend()
#plt.show()

#########################Velocity over###################################
#########################Position start##################################

p = np.zeros(120)

for i in range(120):
    p[i] = p[i-1] + v[i] + 0.5*(a[i]**2)

p_predicted = np.zeros(120)

for i in range(120):
    p_predicted[i] = p_predicted[i-1] + kf_est_velocity[i] + 0.5*(accel_estimate[i]**2)
    
ground_truth_p = p.copy()
##Add noise to the position models
noise_obs_p = np.random.normal(0,687,120)#Q
noise_model_p = np.random.normal(0,2000,120)#R
#simulated_vel = v_predicted + noise_model_velocity
simulated_p = p_predicted
observation_p_i = ground_truth_p + noise_obs_p

for i in range(120):
    observation_p[i] = ground_truth_p[i] + 10*i 
    
cov_sensor_model_p = np.cov(noise_obs_p) #Rt
print(cov_sensor_model_p)
cov_dynamic_model_p = np.cov(noise_model_p) #Qt 


##KF for Velocity
estimated_p = []

##Initial belief params
p_initial=ground_truth[0]
best_p_estimate = 0

##Define model
i=0
while i<120:
    p_initial= simulated_p[i]
    kalman_gain_p = (cov_dynamic_model_p)/(cov_dynamic_model_p + cov_sensor_model_p)
    best_p_estimate = p_initial + kalman_gain_p*(observation_p[i]-simulated_p[i])
    estimated_p.append(best_p_estimate)
    i = i+1
 
kf_est_p = np.array(estimated_p)

plt.plot(range(len(ground_truth_p)), ground_truth_p,label = "Ground Truth position")
plt.plot(range(len(observation_p)), observation_p,label = "Sensor value for position")
plt.plot(range(len(estimated_p)),kf_est_p,label = "Kalman filter position")
#plt.plot(range(len(estimated_p)),p_predicted,label = "Predicted position")
plt.legend()
plt.show()
##Evaluating results

#for i in range(120):
#    MSE_sensor = MSE_sensor + (noise_obs[i] - ground_truth)

MSE_sensor = np.sqrt(0.0083*np.sum(np.abs(observation_p - ground_truth_p)**2))
MSE_kf = np.sqrt(0.0083*np.sum(np.abs(kf_est_p - ground_truth_p)**2))

print("RMSE from raw sensor readings : ", MSE_sensor)
print("RMSE from kalman filter output : ", MSE_kf)