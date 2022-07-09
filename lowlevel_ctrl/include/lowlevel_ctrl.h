
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

#include <sstream>
#include <string>
#include <list>
#include <queue>
#include <mutex> 
#include <thread> 
#include <numeric>
#include <boost/thread/thread.hpp>
#include <eigen3/Eigen/Geometry>

#include <ros/ros.h>
#include <ros/time.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_msgs/Float64.h>
#include <autorally_msgs/chassisCommand.h>
#include <hmcl_msgs/vehicleCmd.h>

#include <dynamic_reconfigure/server.h>
#include <lowlevel_ctrl/testConfig.h>


#include "lowpass_filter.h"
#include "utils.h"



#define PI 3.14159265358979323846264338

class LowlevelCtrl 
{  
private:
ros::NodeHandle nh_ctrl_, nh_signal_;


std::string chassisCmd_topic, imu_topic;

std::mutex mtx_;
ros::Subscriber imuSub, accCmdSub, steerCmdSub, ctleffortSub, odomSub;
ros::Publisher  filt_imu_pub, chassisCmdPub, debugPub, acc_x_pub;

dynamic_reconfigure::Server<lowlevel_ctrl::testConfig> srv;
dynamic_reconfigure::Server<lowlevel_ctrl::testConfig>::CallbackType f;


geometry_msgs::PoseStamped debug_msg;

nav_msgs::Odometry cur_odom;

double throttle_effort;
ros::Time throttle_effort_time;

Butterworth2dFilter imu_x_filter, imu_y_filter, imu_z_filter;
double filter_dt, cutoff_hz;
Butterworth2dFilter roll_filter, pitch_filter, yaw_filter; 
double rpy_dt,rpy_cutoff_hz;  
Butterworth2dFilter pitch_term_filter, rolling_term_filter;
double pitch_term, rolling_term;
double filt_x, filt_y, filt_z;

hmcl_msgs::vehicleCmd vehicle_cmd, prev_vehicle_cmd;
std::array<double,3> cur_rpy;

bool imu_received;
ros::Time state_time, prev_state_time;
double grav_accl = 9.806;
double roll_coef = 0.01035;
double cmd_scale, cmd_offset, brake_scale;
bool enforce_throttle;
double manual_acc_cmd, manual_throttle, manual_brake;


public:
LowlevelCtrl(ros::NodeHandle& nh_ctrl, ros::NodeHandle& nh_traj);
~LowlevelCtrl();

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
void ControlLoop();
void ImuCallback(const sensor_msgs::Imu::ConstPtr& msg);
void accCabllback(const hmcl_msgs::vehicleCmd::ConstPtr& msg);
void controleffortCallback(const std_msgs::Float64::ConstPtr& msg);


void dyn_callback(lowlevel_ctrl::testConfig& config, uint32_t level);



};



