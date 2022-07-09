#!/usr/bin/env python
import rospy


""" ROS node for the MPC GP in 3d offroad environment, to use in the Gazebo simulator and real world experiments.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import json
import time
import threading
import numpy as np
import pandas as pd
import math 
import sys
from std_msgs.msg import Bool, Empty, Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped 
from hmcl_msgs.msg import Lane, Waypoint, vehicleCmd
from visualization_msgs.msg import MarkerArray, Marker
from autorally_msgs.msg import chassisState

from mpc_gp.mpc_model import GPMPCModel
from mpc_gp.mpc_utils import euler_to_quaternion, quaternion_to_euler, unit_quat, get_odom_euler, get_local_vel

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class GPMPCWrapper:
    def __init__(self,environment="gazebo"):
        
        self.n_mpc_nodes = rospy.get_param('~n_nodes', default=40)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)   
        self.model_build_flag = rospy.get_param('~build_flat', default=True)             
        self.dt = self.t_horizon / self.n_mpc_nodes*1.0
        
#################################################################        
        # Initialize GP MPC         
#################################################################
        self.MPCModel = GPMPCModel( model_build = self.model_build_flag,  N = self.n_mpc_nodes, dt = self.dt)
        self.odom_available           = False 
        self.vehicle_status_available = False 
        self.waypoint_available = False 
        
        # Thread for MPC optimization
        self.mpc_thread = threading.Thread()        
        self.chassisState = chassisState()
        self.odom = Odometry()
        self.waypoint = PoseStamped()
        self.debug_msg = PoseStamped()
        self.obs_pose = PoseWithCovarianceStamped()
        # Real world setup
        odom_topic = "/ground_truth/state"
        vehicle_status_topic = "/chassisState"
        control_topic = "/acc_cmd"            
        waypoint_topic = "/move_base_simple/goal"                
        obs_topic = "/initialpose"       
        status_topic = "/is_mpc_busy"
        
        # Publishers
        self.control_pub = rospy.Publisher(control_topic, vehicleCmd, queue_size=1, tcp_nodelay=True)        
        self.mpc_predicted_trj_publisher = rospy.Publisher("/mpc_pred_trajectory", MarkerArray, queue_size=2)
        self.final_ref_publisher = rospy.Publisher("/final_trajectory", MarkerArray, queue_size=2)    
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    
        self.debug_pub = rospy.Publisher("mpc_debug", PoseStamped, queue_size=2)    
        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                        
        self.vehicle_status_sub = rospy.Subscriber(vehicle_status_topic, chassisState, self.vehicle_status_callback)                
        self.waypoint_sub = rospy.Subscriber(waypoint_topic, PoseStamped, self.waypoint_callback)
        self.obs_sub = rospy.Subscriber(obs_topic, PoseWithCovarianceStamped, self.obs_callback)

        # 20Hz control callback 
        self.cmd_timer = rospy.Timer(rospy.Duration(0.1), self.cmd_callback) 
        self.blend_min = 3
        self.blend_max = 5
        self.is_first_mpc = True

        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():
            # Publish if MPC is busy with a current trajectory
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            rate.sleep()

    def obs_callback(self,msg):
        self.obs_pose = msg

    def vehicle_status_callback(self,data):
        if self.vehicle_status_available is False:
            self.vehicle_status_available = True
        self.chassisState = data
        
    def waypoint_callback(self, msg):
        if self.waypoint_available is False:
            self.waypoint_available = True
        self.waypoint = msg

    def run_mpc(self, odom):
        if self.MPCModel is None:
            return
        current_euler = get_odom_euler(self.odom)
        local_vel = get_local_vel(self.odom, is_odom_local_frame = False)
        self.debug_msg.header.stamp = rospy.Time.now()
        self.debug_msg.pose.position.x = local_vel[0]
        self.debug_msg.pose.position.y = local_vel[1]
        self.debug_msg.pose.position.z = local_vel[2]
        self.debug_pub.publish(self.debug_msg)
        
        # xinit = np.transpose(np.array([self.odom.pose.pose.position.x,self.odom.pose.pose.position.y, local_vel[0], current_euler[2], self.chassisState.steering]))
        xinit = np.transpose(np.array([self.odom.pose.pose.position.x,self.odom.pose.pose.position.y, local_vel[0], current_euler[2]]))
        if self.is_first_mpc:
            self.is_first_mpc = False
            # x0i = np.array([0.,0.,self.odom.pose.pose.position.x,self.odom.pose.pose.position.y, 1.0, current_euler[2], self.chassisState.steering])
            x0i = np.array([0.,0.,self.odom.pose.pose.position.x,self.odom.pose.pose.position.y, 1.0, current_euler[2]])
            x0 = np.transpose(np.tile(x0i, (1, self.MPCModel.model.N)))
            problem = {"x0": x0,
                    "xinit": xinit}
        else:
            problem = {"xinit": xinit} 
               
        obstacle_ = np.array([self.obs_pose.pose.pose.position.x, self.obs_pose.pose.pose.position.y])
        goal_ = np.array([self.waypoint.pose.position.x, self.waypoint.pose.position.y])        
        problem["all_parameters"] = np.transpose(np.tile(np.concatenate((goal_,obstacle_)),(1,self.MPCModel.model.N)))        
        output, exitflag, info = self.MPCModel.solver.solve(problem)
        if exitflag != 1: 
            sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n"\
            .format(info.it, info.solvetime))            
            return
            
        
        
        temp = np.zeros((np.max(self.MPCModel.model.nvar), self.MPCModel.model.N))
        for i in range(0, self.MPCModel.model.N):
            temp[:, i] = output['x{0:02d}'.format(i+1)]
        u_pred = temp[0:2, :]
        # x_pred = temp[2:7, :]
        x_pred = temp[2:6, :]
        self.predicted_trj_visualize(x_pred)
        
        ctrl_cmd = vehicleCmd()
        ctrl_cmd.header.stamp = rospy.Time.now()
        ctrl_cmd.acceleration = u_pred[0,0]
        ctrl_cmd.steering =  -1*u_pred[1,0]  #-1*u_pred[1,0]*0.05+self.chassisState.steering
        self.control_pub.publish(ctrl_cmd)

    def cmd_callback(self,timer):
        if self.vehicle_status_available is False:
            rospy.loginfo("Vehicle status is not available yet")
            return
        elif self.odom_available is False:
            rospy.loginfo("Odom is not available yet")
            return
        elif self.waypoint_available is False:
            rospy.loginfo("Waypoints are not available yet")
            return
  

        def _thread_func():
            self.run_mpc(self.odom)            
        self.mpc_thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        self.mpc_thread.start()
        self.mpc_thread.join()
        

    

    #     """
    #     :param odom: message from subscriber.
    #     :type odom: Odometry
    #     :param recording: If False, some messages were skipped between this and last optimization. Don't record any data
    #     during this optimization if in recording mode.
    #     """
    #     if not self.pose_available or not self.vehicle_status_available:
    #         return
    #     if not self.waypoint_available:
    #         return
    #     # Measure time between initial state was checked in and now
    #     dt = odom.header.stamp.to_time() - self.last_update_time

    #     # model_data, x_guess, u_guess = self.set_reference()         --> previous call
    #     if len(x_ref) < self.n_mpc_nodes:
    #         ref = [x_ref[-1], y_ref[-1], psi_ref[-1], 0.0, 0.0, 0.0, 0.0]
    #         u_ref = [0.0, 0.0]   
    #         terminal_point = True               
    #     else:        
    #         ref = np.zeros([7,len(x_ref)])
    #         ref[0] = x_ref
    #         ref[1] = y_ref
    #         ref[2] = psi_ref
    #         ref[3] = vel_ref
    #         ref = ref.transpose()
    #         u_ref = np.zeros((len(vel_ref)-1,2))
    #         terminal_point = False
        
    #     model_data = self.gp_mpc.set_reference(ref,u_ref,terminal_point)        
    
    #     if self.mpc_ready is False: 
    #         return
    #     # Run MPC and publish control
    #     try:
    #         tic = time.time()            
    #         next_control, w_opt, x_opt, self.solver_status = self.gp_mpc.optimize(model_data)
    #         ####################################
    #         if x_opt is not None:                
    #             self.predicted_trj_visualize(x_opt)
    #         ####################################
    #         ## Check whether the predicted trajectory is close to the actual reference trajectory // if not apply auxillary control
    #         self.pred_trj_healthy = self.check_pred_trj(x_opt,ref)
            

    #         if self.solver_status > 0:                                
    #             self.mpc_safe_count = 0
    #             self.reset_mpc_optimizer()                
    #         else: 
    #             self.mpc_safe_count = self.mpc_safe_count + 1
    #         ###### check the number of success rounds of the MPC optimization
    #         if self.mpc_safe_count < self.mpc_safe_count_threshold:
    #             return

    #         if not self.pred_trj_healthy:
    #             return

    #         self.optimization_dt += time.time() - tic
    #         print("MPC thread. Seq: %d. Topt: %.4f" % (odom.header.seq, (time.time() - tic) * 1000))            
    #         control_msg = AckermannDrive()
    #         control_msg = next_control.drive                                                                     
    #         steering_val = max(min(self.steering_rate_max, next_control.drive.steering_angle_velocity), self.steering_rate_min)                        
    #         control_msg.steering_angle = max(min(self.steering_max, steering_val*0.1 + self.steering), self.steering_min)                        
    #         tt_steering = Float64()
    #         tt_steering.data = -1*control_msg.steering_angle            
    #         # control_msg.acceleration = -110.0
    #         self.steering_pub.publish(tt_steering)            
    #         self.control_pub.publish(control_msg)            
            

    #     except KeyError:
    #         self.recording_warmup = True
    #         # Should not happen anymore.
    #         rospy.logwarn("Tried to run an MPC optimization but MPC is not ready yet.")
    #         return

    #     if w_opt is not None:            
    #         self.w_opt = w_opt            
        

    def odom_callback(self, msg):
        """                
        :type msg: PoseStamped
        """              
        if self.odom_available is False:
            self.odom_available = True 
        self.odom = msg        
        
    

    def predicted_trj_visualize(self,predicted_state):        
        marker_refs = MarkerArray() 
        for i in range(len(predicted_state[0,:])):
            marker_ref = Marker()
            marker_ref.header.frame_id = "map"  
            marker_ref.ns = "mpc_ref"+str(i)
            marker_ref.id = i
            marker_ref.type = Marker.ARROW
            marker_ref.action = Marker.ADD                
            marker_ref.pose.position.x = predicted_state[0,i] 
            marker_ref.pose.position.y = predicted_state[1,i]              
            quat_tmp = euler_to_quaternion(0.0, 0.0, predicted_state[3,i])     
            quat_tmp = unit_quat(quat_tmp)                 
            marker_ref.pose.orientation.w = quat_tmp[0]
            marker_ref.pose.orientation.x = quat_tmp[1]
            marker_ref.pose.orientation.y = quat_tmp[2]
            marker_ref.pose.orientation.z = quat_tmp[3]
            marker_ref.color.r, marker_ref.color.g, marker_ref.color.b = (255, 255, 0)
            marker_ref.color.a = 0.5
            marker_ref.scale.x, marker_ref.scale.y, marker_ref.scale.z = (0.6, 0.4, 0.3)
            marker_refs.markers.append(marker_ref)
            i+=1
        self.mpc_predicted_trj_publisher.publish(marker_refs)
        
 
     
        
  
###################################################################################

def main():
    rospy.init_node("mpc_gp")
    env = rospy.get_param('~environment', default='gazebo')
    GPMPCWrapper(env)

if __name__ == "__main__":
    main()




 
    



        # msg.waypoints = msg.waypoints[1:-1]
        # if not self.waypoint_available:
        #     self.waypoint_available = True
        
        # self.x_ref = [msg.waypoints[i].pose.pose.position.x for i in range(len(msg.waypoints))]
        # self.y_ref = [msg.waypoints[i].pose.pose.position.y for i in range(len(msg.waypoints))]                        
        # quat_to_euler_lambda = lambda o: quaternion_to_euler([o[0], o[1], o[2], o[3]])            
        # self.psi_ref = [wrap_to_pi(quat_to_euler_lambda([msg.waypoints[i].pose.pose.orientation.w,msg.waypoints[i].pose.pose.orientation.x,msg.waypoints[i].pose.pose.orientation.y,msg.waypoints[i].pose.pose.orientation.z])[2]) for i in range(len(msg.waypoints))]                                    
        
        # self.vel_ref = [msg.waypoints[i].twist.twist.linear.x for i in range(len(msg.waypoints))]
 
        # while len(self.x_ref) < self.n_mpc_nodes:
        #     self.x_ref.insert(-1,self.x_ref[-1])
        #     self.y_ref.insert(-1,self.y_ref[-1])
        #     self.psi_ref.insert(-1,self.psi_ref[-1])
        #     self.vel_ref.insert(-1,self.vel_ref[-1])

        # self.ref_gen.set_traj(self.x_ref, self.y_ref, self.psi_ref, self.vel_ref)
        
        