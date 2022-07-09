
//   Copyright (c) 2022 Ulsan National Institute of Science and Technology (UNIST)
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

//   Authour : Hojin Lee, hojinlee@unist.ac.kr


#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <cmath>
#include <cstdlib>
#include <chrono>


#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <vector>
#include "lowlevel_ctrl.h"

int test_count = 0;
using namespace std;

LowlevelCtrl::LowlevelCtrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_signal):  
  nh_ctrl_(nh_ctrl),
  nh_signal_(nh_signal),  
  filter_dt(0.005),
  cutoff_hz(10),
  rpy_dt(0.005),
  rpy_cutoff_hz(2),
  cmd_scale(6.1),
  cmd_offset(-0.8),
  brake_scale(1),
  manual_acc_cmd(0.0),
  enforce_throttle(false),
  manual_throttle(0.0),
  manual_brake(0.0)
{
  imu_x_filter.initialize(filter_dt, cutoff_hz);
  imu_y_filter.initialize(filter_dt, cutoff_hz);
  imu_z_filter.initialize(filter_dt, cutoff_hz);

  roll_filter.initialize(rpy_dt, rpy_cutoff_hz);
  pitch_filter.initialize(rpy_dt, rpy_cutoff_hz);
  yaw_filter.initialize(rpy_dt, rpy_cutoff_hz);

  pitch_term_filter.initialize(rpy_dt, rpy_cutoff_hz);
  rolling_term_filter.initialize(rpy_dt, rpy_cutoff_hz);
  
  

  nh_signal_.param<std::string>("chassisCmd_topic", chassisCmd_topic, "/chassisCommand");
  nh_signal_.param<std::string>("imu_topic", imu_topic, "/imu/imu");  
 

  imuSub = nh_signal_.subscribe(imu_topic, 50, &LowlevelCtrl::ImuCallback, this);
  filt_imu_pub  = nh_signal_.advertise<sensor_msgs::Imu>("filtered_imu", 2);      
  chassisCmdPub  = nh_ctrl.advertise<autorally_msgs::chassisCommand>(chassisCmd_topic, 2);   

  debugPub  = nh_ctrl.advertise<geometry_msgs::PoseStamped>("/lowlevel_debug", 2);   
  
  acc_x_pub  = nh_ctrl.advertise<std_msgs::Float64>("/state", 2);   
  ctleffortSub = nh_signal_.subscribe("/control_effort", 50, &LowlevelCtrl::controleffortCallback, this);   


  accCmdSub = nh_signal_.subscribe("/acc_cmd", 50, &LowlevelCtrl::accCabllback, this);   
  odomSub = nh_signal_.subscribe("/ground_truth/state", 50, &LowlevelCtrl::odomCallback, this);   


  boost::thread ControlLoopHandler(&LowlevelCtrl::ControlLoop,this);   
  ROS_INFO("Init Lowlevel Controller");
  
  f = boost::bind(&LowlevelCtrl::dyn_callback,this, _1, _2);
	srv.setCallback(f);

}

LowlevelCtrl::~LowlevelCtrl()
{}

void LowlevelCtrl::odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
  cur_odom = *msg;
  
     tf::Quaternion q_(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
    tf::Matrix3x3 m(q_);
    m.getRPY(cur_rpy[0], cur_rpy[1], cur_rpy[2]);
    cur_rpy[0] = -cur_rpy[0];
    cur_rpy[1] = -cur_rpy[1];
    cur_rpy[2] = cur_rpy[2];

    cur_rpy[0] = roll_filter.filter(cur_rpy[0]);
    cur_rpy[1] = pitch_filter.filter(cur_rpy[1]);
    cur_rpy[2] = yaw_filter.filter(cur_rpy[2]);

    debug_msg.header = msg->header;
    debug_msg.pose.position.x = cur_rpy[0];
    debug_msg.pose.position.y = cur_rpy[1];
    debug_msg.pose.position.z = cur_rpy[2];
    debugPub.publish(debug_msg);
}

void LowlevelCtrl::controleffortCallback(const std_msgs::Float64::ConstPtr& msg){
  
  throttle_effort_time = ros::Time::now();
  throttle_effort = msg->data;
}

void LowlevelCtrl::ImuCallback(const sensor_msgs::Imu::ConstPtr& msg){  
  if(!imu_received){
    imu_received = true;
  }
  sensor_msgs::Imu filtered_imu_msg;
  filtered_imu_msg = *msg;
  
   // x - sin(pitch)*grav_accl 
  double accel_x_wihtout_gravity = msg->linear_acceleration.x -sin(cur_rpy[1])*grav_accl;
  // y + cos(pitch)sin(roll)*grav_accl
  double accel_y_wihtout_gravity = msg->linear_acceleration.y +cos(cur_rpy[1])*sin(cur_rpy[0])*grav_accl;
  // z - cos(pitch)cos(roll)*grav_accl
  double accel_z_wihtout_gravity = msg->linear_acceleration.z -cos(cur_rpy[1])*cos(cur_rpy[0])*grav_accl;
  filt_x = imu_x_filter.filter(accel_x_wihtout_gravity);
  filt_y = imu_y_filter.filter(accel_y_wihtout_gravity);
  filt_z = imu_z_filter.filter(accel_z_wihtout_gravity);
  filtered_imu_msg.linear_acceleration.x = filt_x;
  filtered_imu_msg.linear_acceleration.y = filt_y;
  filtered_imu_msg.linear_acceleration.z = filt_z;
  filt_imu_pub.publish(filtered_imu_msg);
  std_msgs::Float64 acc_x_msg;
  acc_x_msg.data = filt_x;
  acc_x_pub.publish(acc_x_msg);
}




void LowlevelCtrl::accCabllback(const hmcl_msgs::vehicleCmd::ConstPtr& msg){    
  vehicle_cmd.header = msg->header;
  vehicle_cmd.header.stamp = ros::Time::now();
  vehicle_cmd.acceleration = msg->acceleration;
  vehicle_cmd.steering     = msg->steering;
  
}



void LowlevelCtrl::ControlLoop()
{   double cmd_rate;
    ros::Rate loop_rate(50); // rate  
    while (ros::ok()){         
        auto start = std::chrono::steady_clock::now();        
        
        ///////////////////////////////////////////////////////

        // Prepare current State for state feedback control 
        // if(!stateSetup()){
        //   ROS_WARN("Path is not close to the current position");
        //    loop_rate.sleep();
        //   continue;
        // }
      if(imu_received){
        
        autorally_msgs::chassisCommand chassis_cmd;
        /////////////////// Control with Mapping //////////////////
        chassis_cmd.header = vehicle_cmd.header;      
                          
        double bias = 1.0;
        
        double brake_bias = 1.102;
        double throttle_cmd = 0.0;
        double brake_cmd = 0.0;
        
        double compensated_pitch;         
        if (cur_rpy[1] >= 0 && cur_rpy[1] < 10*PI/180.0){
          compensated_pitch = 0.0;
        }
        // else if( cur_rpy[1] < 0 && cur_rpy[1] > -10*PI/180.0){
        //   compensated_pitch = 0.0;
        // }
        else{
          compensated_pitch = cur_rpy[1];
        }        
        pitch_term = grav_accl*sin(compensated_pitch);                        
        pitch_term = pitch_term_filter.filter(pitch_term);   
        
        rolling_term = roll_coef*grav_accl*cos(compensated_pitch);
        rolling_term = rolling_term_filter.filter(rolling_term);
        // throttle_cmd = (vehicle_cmd.acceleration + roll_coef*grav_accl*cos(compensated_pitch)+grav_accl*sin(compensated_pitch))/scale;
        debug_msg.pose.orientation.x =  pitch_term;
        debug_msg.pose.orientation.y =  rolling_term;
        
        double diff_time = fabs((vehicle_cmd.header.stamp - ros::Time::now()).toSec());        
        double local_speed = sqrt(pow(cur_odom.twist.twist.linear.x,2)+pow(cur_odom.twist.twist.linear.y,2)+pow(cur_odom.twist.twist.linear.z,2));
        // scale = 10.0;
        // scale = fabs(1.319*log(vehicle_cmd.acceleration)+3.649);
        // scale = std::min(std::max(scale,1.0),10.0);
        // if(cmd_scale  == 0.0){
        // cmd_scale = 1e-5;
        // }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // cmd_scale * (throttle-brake+cmd_offset) =  vehicle_cmd.acceleration*vehicle_mass + rolling_term + pitch_term 
        // throttle - brake = vehicle_cmd.acceleration*vehicle_mass + rolling_term + pitch_term  - cmd_offset/cmd_scale; 
        // throttle - brake {0,0} = (0.0 +rolling_term + pitch_term)/cmd_scale - cmd_off/cmd_scale;        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // if(diff_time < 0.5){         
            throttle_cmd = (manual_acc_cmd + rolling_term+pitch_term)/cmd_scale - cmd_offset/cmd_scale;                     
            // if(local_speed < 0.1 && vehicle_cmd.acceleration <=0){
              if(local_speed < 0.1 && manual_acc_cmd <=0){
              throttle_cmd = 0.0;              
            }          
          if ( throttle_cmd < 0.0){
            brake_cmd = -throttle_cmd;
          }
          throttle_cmd = std::max(std::min(throttle_cmd,1.0),0.0);          
          brake_cmd = std::max(std::min(brake_cmd,1.0),0.0);          
        // }
        /////////////////// Control with Mapping END //////////////////

        /////////////////// Control with PID node   ///////////////////////////////////////
        // double throttle_cmd = 0.0;
        // double diff_time = fabs((throttle_effort_time- ros::Time::now()).toSec());        
        // if(diff_time < 0.1){              
        //   throttle_cmd = throttle_effort;
        //   throttle_cmd = std::min(throttle_cmd,1.0);
        //   throttle_cmd = std::max(throttle_cmd,0.0);
        // }      
        /////////////////// Control with PID node End --> not smooth...  //////////////////
        debug_msg.pose.orientation.z =  throttle_cmd;        
        debug_msg.pose.orientation.w =  brake_cmd;
        debugPub.publish(debug_msg); 
        if(enforce_throttle){
          // throttle_cmd = manual_throttle;
        chassis_cmd.throttle = throttle_cmd;        
        chassis_cmd.frontBrake =brake_cmd;
        chassis_cmd.steering =vehicle_cmd.steering/(25*PI/180.0);               
        chassisCmdPub.publish(chassis_cmd);
        }else{
          throttle_cmd = 0.0;
          brake_cmd = 1.0;     
        chassis_cmd.throttle = throttle_cmd;        
        chassis_cmd.frontBrake =brake_cmd;
        chassis_cmd.steering =vehicle_cmd.steering/(25*PI/180.0);               
        chassisCmdPub.publish(chassis_cmd);
        }
        
      }


     auto end = std::chrono::steady_clock::now();     
     loop_rate.sleep();
     std::chrono::duration<double> elapsed_seconds = end-start;
     if ( elapsed_seconds.count() > 1/cmd_rate){
       ROS_ERROR("computing control gain takes too much time");
       std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
     }
      
    }
}

void LowlevelCtrl::dyn_callback(lowlevel_ctrl::testConfig &config, uint32_t level)
{  
        enforce_throttle = config.enforce_throttle;
        // cmd_scale = config.scale;
        // cmd_offset = config.offset;
        manual_acc_cmd = config.manual_acc_cmd;
        manual_throttle=config.manual_throttle; 
        double scale = 6.1;        
        double offset = -0.8;
        
    
}






int main (int argc, char** argv)
{
  ros::init(argc, argv, "LowlevelCtrl");
  
  ros::NodeHandle nh_ctrl, nh_signal;
  LowlevelCtrl LowlevelCtrl(nh_ctrl, nh_signal);

  ros::CallbackQueue callback_queue_ctrl, callback_queue_signal;
  nh_ctrl.setCallbackQueue(&callback_queue_ctrl);
  nh_signal.setCallbackQueue(&callback_queue_signal);
  

  std::thread spinner_thread_ctrl([&callback_queue_ctrl]() {
    ros::SingleThreadedSpinner spinner_ctrl;
    spinner_ctrl.spin(&callback_queue_ctrl);
  });

  std::thread spinner_thread_signal([&callback_queue_signal]() {
    ros::SingleThreadedSpinner spinner_signal;
    spinner_signal.spin(&callback_queue_signal);
  });

 

    ros::spin();

    spinner_thread_ctrl.join();
    spinner_thread_signal.join();


  return 0;

}
