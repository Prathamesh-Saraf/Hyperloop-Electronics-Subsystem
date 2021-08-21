import numpy as np
import matplotlib.pyplot as plt

##Input step function
#ground_truth = 10*np.ones(100)
#ground_truth[20:] = ground_truth[20:] - 10
#ground_truth[80:] = ground_truth[80:] - 10

a = np.logspace(7,0,num=120)
a = 2.6*a/100000 
a[50:90] = a[50:90] - 35

ground_truth = a.copy()
ground_truth_1 = a.copy()
ground_truth_2 = a.copy()
noise_obs = np.random.normal(0,10,120)#Q
noise_model = np.random.normal(0,5,120)#R
simulated_accel = ground_truth_1 + noise_model
observation = ground_truth_2 + noise_obs
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

#print(estimated_acceleration) #Debug step
#Plotting the results
plt.plot(range(len(ground_truth)), ground_truth,label = "Ground Truth")
plt.plot(range(len(observation)), observation,label = "Sensor value")
plt.plot(range(len(estimated_acceleration)),estimated_acceleration,label = "Kalman filter")
plt.legend()
plt.show()


##Evaluating results
kf_est = np.array(estimated_acceleration)

#for i in range(120):
#    MSE_sensor = MSE_sensor + (noise_obs[i] - ground_truth)

MSE_sensor = np.sqrt(0.0083*np.sum(np.abs(noise_obs - ground_truth)**2))
MSE_kf = np.sqrt(0.0083*np.sum(np.abs(kf_est - ground_truth)**2))

print("RMSE from raw sensor readings : ", MSE_sensor)
print("RMSE from kalman filter output : ", MSE_kf)