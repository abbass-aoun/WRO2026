


#start the robot
from concurrent.futures import wait


def Start():
    
    
#see if the start button is pressed
def check_start():  


#read from sensors and update the EKF
def read_sensors_and_update_ekf():
    

#move in a straight line , using the EKF to correct heading
def run_ekf_straight(sensor readingsd, control): out :positojn
    
    
#move in a corner , using the EKF to correct heading    
def run_ekf_corner():
    
            
#when seeing a pillar, use the EKF to correct heading and move towards the pillar
def run_ekf_pillar():
        
    
#bezier curve to move towards the pillar, also take a distance from the pillar to avoid collision.  
def calculate_trajectory_to_pillar(): 
    
#bezier curve to move , but without a pillar 
def calculate_trajectory_no_pillar():       
    
#bezier curve to move towards the corner, but without a pillar    
def calculate_trajectory_to_corner(): 
    
    
def see_wall():
    
#This method calculates the nearest distance from the robot to the curve
def compute_tracking_error():#used in get tarjectory methods
    
#this method will update the local reference to the current position
def  update_reference():#transformation matrix
    
#This method should drive the robot 1 step according to the saved trajectory
def take_step():
       
       
def see_pillar():
    
#this is the a cv part,maybe more than one function.
def process_data(vision):           

#this is to count, sections, laps and corners.
def add_Section():    
    
#decides if the robot should stop in this section or not, based on the number of sections and laps
def stop_in_this_section():    
    
  
def parking():
    
#if we start with the parking lot, we need to drive out

def exit_parking_lot():
    

#the trick after lap 2
def special_trick():
    
    
    
    
    
    
    
    
    
    
    
    
    

while not start_button:
    time.sleep(0.0001)
    check_start()

print("button pressed. The main loop will start now")

error_to_track_threshold=5
#the main loop
while True:     #TODO COMPATIBLE...
    
    #Process sensor and camera data
    process_data()
    
    drive_direction,drive_speed,steering_angle = take_step()# The robot should move a step according to the saved trajectory
    drive_speed=-1*drive_speed if drive_direction=="b" else drive_speed
    current_position=EKF_estimator.update(current_position,ms,drive_speed,steering_angle,RWR_encoder,RWL_encoder,steer_encoder,imu_z)

    #check if we are in the parking:
    if parking_exists and distance_sensors[1]<10 and distance_sensors[4]<10 and lap_nb==1 and section_nb==1:
        exit_parking()
        continue
    #check if it is the first time to assign trajectory
    if trajectory is None:# 
        if(not pillars): #if the list of the pillar is empty
            if (not in_corner):
                calculate_trajectory_no_pillar()
            else:
                calculate_trajectory_to_corner()
            update_reference()
        else:
            calculate_trajectory-to_pillar()
            update_reference()
    else: #if there is already a trajectory decide wether to create new one or stick with the old depending on the error
         error_to_track=compute_tracking_error()
         if error_to_track>error_to_track_threshold:
            if(not pillars):
              calculate_trajectory_no_pillar()

            else:
                calculate_trajectory()



   
    



    
    
    #check if we entered or exited the corner and update the direction
    DEBOUNCE_S = 2.0   # match your stated rule

now = time.time()
    if now - last_line_time > DEBOUNCE_S:

    # Lock in direction ONCE, on the very first line seen
        if direction is None:
            direction = CCW if color_sensor == "blue" else CW
            turn_start_color = "blue" if direction == CCW else "orange"
            turn_stop_color  = "orange" if direction == CCW else "blue"

        if not in_corner and color_sensor == turn_start_color:
            in_corner = True
            last_line_time = now

        elif in_corner and color_sensor == turn_stop_color:
            in_corner = False
            add_Section()
            last_line_time = now

    #stop in the reached section if this is the final section
        if True:
            if(final_section and parking_exists):
            park()
            elif(final_section):
            stop_in_this_section()
            break
        
    #update the time
    time_t=time.time

    #exit the loop if q is pressed on the keyboard
    if keyboard.is_pressed('q'):
        print("Quitting...")
        break