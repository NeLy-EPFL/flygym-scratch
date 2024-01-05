import numpy as np
from tqdm import trange
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt

from flygym.mujoco import Parameters, NeuroMechFly
from flygym.mujoco.examples.common import PreprogrammedSteps
from flygym.mujoco.examples.cpg_controller import CPGNetwork
from flygym.mujoco.examples.turning_controller import HybridTurningNMF
from keras.models import Model,load_model
from random_walk import generate_random_walk
import joblib

#Class for our controller 

class Path_NMF(HybridTurningNMF):
    def __init__(
        self,
        preprogrammed_steps=None,
        amplitude_range=(-1.2, 1.2),
        draw_corrections=False,
        rw_step = 20000, #min 4000
        #PID parameter, only proportionnal needed but can be upgraded
        kp = 100,
        ki = 0,
        kd = 0,
        previous_error = 0,
        integral= 0,
        PID_treshold= 10, #deg
        output_treshold= 40,
        #Load models
        drive_model = joblib.load('drive_regression_model.joblib'),
        step_model = joblib.load('step_regression_model.joblib'),
        pos_x_model = load_model('lstm_final_x.keras'),
        pos_y_model = load_model('lstm_final_y.keras'),
        #Store data
        return_vector_hist = [],
        heading_truth_hist = [],
        heading_calculated_hist = [],
        position_truth_hist = [],
        drive_hist = [],
        position_estimated = [],
        seed=10,
        turnings = None,
        rw_ended = False,
        max_positive = 1.2,
        min_negative = -1.2,
        group_size = 10,
        w = 4000,
        reached = False,
        **kwargs,

    ):
        # Initialize core NMF simulation
        super().__init__(**kwargs)

        if preprogrammed_steps is None:
            preprogrammed_steps = PreprogrammedSteps()
        self.preprogrammed_steps = preprogrammed_steps
        self.amplitude_range = amplitude_range
        self.draw_corrections = draw_corrections
        #Number of step of RW
        self.rw_step = rw_step
        #Random walk
        self.turnings = generate_random_walk(rw_step)
        self.rw_ended = rw_ended
        self.heading_truth_hist = heading_truth_hist
        self.return_vector_hist = return_vector_hist
        self.heading_calculated_hist = heading_calculated_hist
        self.position_truth_hist = position_truth_hist
        self.drive_hist = drive_hist
        self.integral = integral
        self.max_positive = max_positive
        self.min_negative = min_negative
        self.output_treshold = output_treshold
        self.PID_treshold = PID_treshold
        self.drive_model = drive_model
        self.step_model = step_model
        self.previous_error = previous_error
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.w = w
        self.group_size = group_size
        self.position_estimated = position_estimated
        self.pos_x_model = pos_x_model
        self.pos_y_model = pos_y_model
        self.reached = reached

    def reset(self, seed=None, init_phases=None, init_magnitudes=None, **kwargs):

        obs, info = super().reset(seed=seed)
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.reset(init_phases, init_magnitudes)
        self.retraction_correction = np.zeros(6)
        self.stumbling_correction = np.zeros(6)
        return obs, info
    
    def custom_mean(self,x):
        return np.nanmean(x[::self.group_size])

    def calculate_rolling_mean(self,drive_diff):
        "Calculate rolling mean for an array"
        drive_diff_mean = np.full_like(drive_diff, np.nan)
        for i in range(len(drive_diff)):
            start_idx = max(0, i - self.w + 1)
            end_idx = i + 1
            window_data = drive_diff[start_idx:end_idx]
            drive_diff_mean[i] = self.custom_mean(window_data)
        return drive_diff_mean
    
    def angle_to_vector(self,angle_degrees):
        """ Convert an angle to a 2D vector """
        angle_degians = np.radians(angle_degrees)
        vector = np.array([np.sin(angle_degians), np.cos(angle_degians)])
        return vector

    
    def heading_regression(self):
        """ Function for evaluating the angle regression
        Note:  an important part of the function is commented since 
        we could not use the prediction due to boundary issues but it is still functional"""

        #No mouvement in the beginning
        i =  np.shape(self.drive_hist)[0]
        if  i <1000:
            return 90
        """ 
        drive = np.array(self.drive_hist)
        drive_diff = drive[:,0] - drive[:,1]
        drive_diff = np.array(drive_diff)
        i =  np.shape(self.drive_hist)[0] 
        return_points = self.return_vector_hist
        return_points = np.array(return_points)
        elif i == 1000:
            drive_diff_mean = self.calculate_rolling_mean(drive_diff)
            drive_diff_mean = drive_diff_mean.reshape(-1, 1)
            self.return_vector_hist = (90 - self.drive_model.predict(drive_diff_mean)).tolist()
            print(np.shape(self.return_vector_hist),np.shape(drive_diff),np.shape(drive_diff_mean))
            print(np.shape((self.drive_model.predict(drive_diff_mean)).tolist()))

        if (i %100) == 0 :
            drive_diff_mean = self.calculate_rolling_mean(drive_diff)
            drive_diff_mean = drive_diff_mean.reshape(-1, 1)
            results = (self.drive_model.predict(drive_diff_mean)[-1])
            results = 90 - results
        else:
           return np.array(return_points[-1]) """
        #Use simulation data for angle ground truth instead
        array = np.array(self.heading_truth_hist)
        y = array[-1,1]
        x = array[-1,0]
        angle = np.arctan2(y,x)
        angle = np.degrees(angle)
        return angle
    
    def PID_return(self):
        """ PID controller computing the retour angle depending on the 
            current angle and position
            Args: None
            Output: action: the array for the step() to reach the goal """

        timestep = 1e-4

        #get current orientation
        current_heading = self.return_vector_hist[-1]

        #get estimated position and deduce the angle
        pos = np.array(self.position_estimated[-1])
        theta = np.arctan2(pos[1],pos[0])
        theta = np.degrees(theta)
        
        #Put the heading in the good reference
        angle_deg = current_heading
        angle_deg = angle_deg + 180
        factor = angle_deg //180
        if factor%2 == 0: 
            angle_deg = angle_deg - factor*180
        else:
            angle_deg = -180 + (angle_deg - factor*180)
       
       #Combine the position angle with the fly orientation and 
       #replace it in the good frame
        angle_deg = 180 - angle_deg + theta
        angle_deg = angle_deg + 180

        factor = angle_deg //180

        if factor%2 == 0: 
            angle_deg = angle_deg - factor*180
        else:
            angle_deg = -180 + (angle_deg - factor*180)

        angle_deg = - angle_deg

        #start of PID controller
        proportional = angle_deg

        self.integral = self.integral + angle_deg*timestep
        derivative = (angle_deg - self.previous_error)/timestep
        self.previous_error = angle_deg
        output = self.kp*proportional + self.ki*self.integral + self.kd*derivative

        #go straight if aligned with the angle
        if np.abs(angle_deg)<self.PID_treshold:
            action_x = 1.
            action_y = 1.

        #else deduce bounded signal for action
        elif output>0  :
            if output>self.output_treshold:
                output= self.output_treshold
            action_x = self.max_positive*output/self.output_treshold
            action_y = self.min_negative*output/self.output_treshold
        elif output<0:
            if output<-self.output_treshold:
                output =-self.output_treshold
            action_x = self.max_positive*output/self.output_treshold
            action_y = self.min_negative*output/self.output_treshold
        return np.array([action_x,action_y])

    def transform_point(self,fly_position_local, fly_orientation_local, pos_fly_frame):

        """ This function change coordinate between from body frame to local frame. 
            It is the complementary to the function in the lstm_position.ipynb file.
            Args: fly_position_local: [x,y] array, fly position in local frame
                  fly_orientation_local: angle in degree, fly orientation in local frame
                  pos_fly_frame: [x,y]: position in body frame of the fly
            Return: point_local_frame: [x,y] array with the new coordinates in local frame """

        orientation_rad = np.radians(fly_orientation_local - 90)
        rotation_matrix = np.array([[np.cos(orientation_rad), -np.sin(orientation_rad)],
                                    [np.sin(orientation_rad), np.cos(orientation_rad)]])
        pos_fly_frame = np.array(pos_fly_frame)
        point_local_frame = np.dot(rotation_matrix, pos_fly_frame) + fly_position_local

        return np.array(point_local_frame)

    def estimation_position(self):
        """ Estimate the fly position based on the drive 
            Return: [x,y], fly position in local frame"""

        i =  np.shape(self.drive_hist)[0]
        #No mouvement at first
        if i < 3000: return [0,0]

        #every 100steps, update measurement
        if i%100 == 0:
            drive = np.array(self.drive_hist)
            #calculate based on the 3000 last drive
            drive = drive[-3000:,:]
            drive = drive.reshape((1, 3000, 2))

            #predict the x and y differences
            delta_x = self.pos_x_model.predict(drive,verbose =0 )
            delta_y = self.pos_y_model.predict(drive,verbose = 0)
            delta_x = delta_x[0][0]
            #Factor to solve scale problem
            delta_y = 0.5*delta_y[0][0]

            #Manual check: symetric drive = no mouvement
            last_drive = np.array(self.drive_hist[-1])
            if last_drive[0] == -last_drive[1]:
                delta_x = 0
                delta_y = 0
           
            # Put the delta position into local frame coordinates
            # Integer the position based on the -3000 step position
            position = np.array(self.position_estimated)[-3000+1,:]
            orientation = np.mean(self.return_vector_hist[-3000:-2000])
            pos_diff = np.array([delta_x,delta_y])
            position_local = self.transform_point(position,orientation,pos_diff)
        else:
            position_local = np.array(self.position_estimated)[-1,:]

        return position_local

    def step(self,i):
        
        action = np.array([0,0])
        #Dont move in the beginning( for position prediction)
        if i < 3000: action = np.array([0,0])
        # Random walk
        elif (i >= 3000) and (i < self.rw_step + 3000): action  = self.turnings[i-3000]
        elif (i == (self.rw_step + 3000)): self.rw_ended = True

        #start PID
        elif (self.rw_ended): action = self.PID_return()

        #Check for reached goal
        if i > 0 : pos = np.array(self.position_estimated[-1])
        else: pos = np.array([0,0])
        if i >(3500 + self.rw_step + 500) and np.abs(pos[0]) < 1 and np.abs(pos[0]) < 1:
            action = np.array([0,0])
            self.reached = True
        if self.reached:
            action = np.array([0,0])

        obs, reward, terminated, truncated, info = super().step(action)

        #store variable
        self.drive_hist.append(action)
        self.return_vector_hist.append(self.heading_regression())
        self.position_estimated.append(self.estimation_position())
        angle = np.array(self.return_vector_hist)

        #Check for continuity
        for i in range(len(angle)):
            if (angle[i] - angle[i-1]) < -180 and i != 0:
                angle[i] += 360
            elif (angle[i] - angle[i-1]) > 180 and i != 0:
                angle[i]-=360 

        self.heading_truth_hist.append(obs["fly_orientation"][:2])
        self.position_truth_hist.append(obs["fly"][0,:2])


        return obs, reward, terminated, truncated, info


if __name__ == "__main__":

    run_time = 6.
    timestep = 1e-4
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    sim_params = Parameters(
        timestep=1e-4,
        render_mode="saved",
        render_camera="Animat/camera_top",
        render_playspeed=0.1,
        enable_adhesion=True,
        draw_adhesion=True,
        actuator_kp=20,
    )

    nmf = Path_NMF(
        sim_params=sim_params,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(0, 0, 0.2),
    )
    #check_env(nmf)

    obs, info = nmf.reset()
    for i in trange(int(run_time / nmf.sim_params.timestep)):
        obs, reward, terminated, truncated, info = nmf.step(i)
        nmf.render()
    nmf.save_video("hybrid_turning.mp4")

    #Plot results


    #Plot the the position
    position_estimated = np.array(nmf.position_estimated)
    truth_position = np.array(nmf.position_truth_hist)

    timestamps = np.linspace(0,run_time,len(position_estimated))

    plt.figure(figsize=(12, 20))
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, position_estimated[:,0],label= 'Estimated')
    plt.plot(timestamps, truth_position[:,0],label = "Ground Truth")
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [mm]')
    plt.title('X position comparison')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(timestamps, position_estimated[:,1],label= 'Estimated')
    plt.plot(timestamps, truth_position[:,1],label = "Ground Truth")
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [mm]')
    plt.title('Y position comparison')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(position_estimated[:,0],position_estimated[:,1],label= 'Estimated')
    plt.plot(truth_position[:,0], truth_position[:,1],label = "Ground Truth")
    plt.xlabel('X Distance [mm]')
    plt.ylabel('Y Distance [mm]')
    plt.title('Trajectory comparison')
    plt.legend()

    plt.savefig('Position_full.png')
    plt.close()

    plt.plot(truth_position[:,0], truth_position[:,1],label = "Ground Truth")
    plt.xlabel('X Distance [mm]')
    plt.ylabel('Y Distance [mm]')
    plt.title('Trajectory of the ground truth')
    plt.legend()

    plt.savefig('Position_truth_only.png')
    plt.close()

    
    plt.plot(position_estimated[:,0], position_estimated[:,1],label = "Estimation")
    plt.xlabel('X Distance [mm]')
    plt.ylabel('Y Distance [mm]')
    plt.title('Trajectory of the estimation')
    plt.legend()

    plt.savefig('Position_estimate_only.png')
    plt.close()


    #Do and plot angle estimation
    drive = np.array(nmf.drive_hist)
    drive_diff = drive[:,0] - drive[:,1]
    drive_diff = np.array(drive_diff)

    drive_diff_mean = nmf.calculate_rolling_mean(drive_diff)
    drive_diff_mean = drive_diff_mean.reshape(-1, 1)
    results = (nmf.drive_model.predict(drive_diff_mean))

    heading_truth = np.array(nmf.heading_truth_hist)
    heading_truth = np.degrees(np.arctan2(heading_truth[:,1],heading_truth[:,0]))

    #linearize over one rotation
    for i in range(len(heading_truth)):
            if (heading_truth[i] - heading_truth[i-1]) < -180 and i != 0:
                heading_truth[i] += 360
            elif (heading_truth[i] - heading_truth[i-1]) > 180 and i != 0:
                heading_truth[i]-=360

    timestamps = np.linspace(0,run_time,len(heading_truth))   
    plt.plot(timestamps,heading_truth,label= 'Ground truth' )
    plt.plot(timestamps,results,label = 'Estimated')

    plt.title('Heading comparison')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [deg]')

    plt.savefig("heading.png")



    