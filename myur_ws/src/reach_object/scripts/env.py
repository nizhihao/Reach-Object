#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np 
import rospy
import actionlib
from control_msgs.msg import *
from trajectory_msgs.msg import *
from sensor_msgs.msg import JointState
from tf import TransformListener
from math import pi 
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import sys, tf
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from copy import deepcopy

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
DURATION = 0.01
GOAL = [0.5,0.5,1.0,0,1.57,0]
INIT = [1.0357,-0.5524,1.5880,-2.8308,-1.5880,0]


class Ur5(object):
    get_counter = 0

    def __init__(self, init_joints=INIT, goal_pose=GOAL, duration=DURATION):
        rospy.init_node('ur5_env', anonymous=True)
        parameters = rospy.get_param(None)
        index = str(parameters).find('prefix')
        if (index > 0):
            prefix = str(parameters)[index+len("prefix': '"):(index+len("prefix': '")+str(parameters)[index+len("prefix': '"):-1].find("'"))]
            for i, name in enumerate(JOINT_NAMES):
                JOINT_NAMES[i] = prefix + name
        
        self.client = actionlib.SimpleActionClient('/arm_controller/follow_joint_trajectory',
                                                                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        self.initial= FollowJointTrajectoryGoal()
        self.initial.trajectory = JointTrajectory()
        self.initial.trajectory.joint_names = JOINT_NAMES
        self.current_joints = init_joints
        self.initial.trajectory.points = [JointTrajectoryPoint(positions=INIT, velocities=[0]*6, 
                                                                        time_from_start=rospy.Duration(duration))]                                                                
        self.tf = TransformListener()                            
        self.goal_pose = np.array(goal_pose)
        self.base_pos = self.get_pos(link_name='base_link')
        self.duration = duration
        self.state_dim = 10
        self.action_dim = 5
    
    def step(self,action):
        #Execute action 
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        #set action joint limit
        action_ = np.concatenate((action,0),axis=None)
        self.current_joints += action_
        action_sent = np.zeros(6)
        for i in range(5):
            if i == 3:
                self.current_joints[i] += np.pi
                if self.current_joints[i] > np.pi:
                    action_sent[i] = self.current_joints[i] % -np.pi
                elif self.current_joints[i] < -np.pi:
                    action_sent[i] = self.current_joints[i] % np.pi
                else:
                    action_sent[i] = self.current_joints[i]
                self.current_joints[i] -= np.pi
                action_sent[i] -= np.pi
            else:
                if self.current_joints[i] > np.pi:
                    action_sent[i] = self.current_joints[i] % -np.pi
                elif self.current_joints[i] < -np.pi:
                    action_sent[i] = self.current_joints[i] % np.pi
                else:
                    action_sent[i] = self.current_joints[i]

        goal.trajectory.points = [JointTrajectoryPoint(positions=action_sent, velocities=[0]*6, 
                                                                    time_from_start=rospy.Duration(self.duration))]
        self.client.send_goal(goal)
        self.client.wait_for_result()
        position, rpy = self.get_pos(link_name='ee_link')

        state = self.get_state(action,position)
        reward, terminal = self.get_reward(position,rpy,action)

        return state, reward, terminal


    def reset(self):
        self.current_joints = INIT
        self.client.send_goal(self.initial)
        self.client.wait_for_result()
        self.target_generate()
        position = self.get_pos(link_name='ee_link')[0]

        return self.get_state([0,0,0,0,0],position)


    def get_pos(self,link_name='ee_link',ref_link='world'):
        position = None
        while position is None:
            try:
                if self.tf.frameExists('wrist_2_link') and self.tf.frameExists(link_name):
                    t = self.tf.getLatestCommonTime(ref_link, link_name)
                    position, quaternion = self.tf.lookupTransform(ref_link, link_name, t)
                    rpy = euler_from_quaternion(quaternion)
            except:
                print 'fail to get data from tf'

        return np.array(position), np.array(rpy)


    def get_state(self,action,position):
        #x, y, z of goal
        goal_pose = self.goal_pose[:3]
        pose_6 = position - goal_pose
        dis_6 = np.linalg.norm(pose_6)
        in_point = 1 if self.get_counter > 0 else 0
        
        state = np.concatenate((pose_6,dis_6,action,in_point),axis=None)
        return state


    def get_reward(self,pos,rpy,action):
        threshold = 2
        t = False
        #Compute reward based on distance
        dis = np.linalg.norm(self.goal_pose[:3]-pos)
        #add regularization term
        # reward = -0.1 * dis - 0.01 * np.linalg.norm(action)
        reward = -0.1 * dis
        #compute reward based on rotation
        dis_a = np.linalg.norm(1.57-rpy[1])
        r_a = -0.5 * dis_a 

        if dis < 0.1:
            reward += 1 + r_a
            print 'reach distance'
            if dis_a < 0.2:
                reward += 2
                print 'reach rotation'
                self.get_counter += 1
            else:
                self.get_counter = 0

            if self.get_counter > threshold:
                reward += 10 
                t = True
                self.get_counter = 0
                print 'successfully complete task'
                print '############################'
        
        return reward, t


    def target_vis(self,goal):
        rospy.wait_for_service("gazebo/delete_model")
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        
        s = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
        
        orient = Quaternion(*tf.transformations.quaternion_from_euler(0, 1.57, 0))
        origin_pose = Pose(Point(goal[0],goal[1],goal[2]), orient)
        with open('/home/user/myur_ws/src/ur_robotiq/ur_robotiq_gazebo/meshes/cube/model.sdf',"r") as f:
            reel_xml = f.read()
        
        for row in [1]:
            for	col in xrange(1):
                reel_name = "reel_%d_%d" % (row,col)
                delete_model(reel_name)
                pose = deepcopy(origin_pose)
                pose.position.x = origin_pose.position.x #- 3.5 * unit + col * unit
                pose.position.y = origin_pose.position.y #- 3.5 * unit + row * unit
                pose.position.z = origin_pose.position.z
                s(reel_name, reel_xml, "", pose, "world")


    def target_generate(self):
        rand_x, rand_y, rand_z= np.random.uniform(-0.2,0.1), np.random.uniform(-0.2,0.1), np.random.uniform(-0.5,-0.1)
        self.goal_pose = np.array(GOAL)
        self.goal_pose[0] += rand_x
        self.goal_pose[1] += rand_y
        self.goal_pose[2] += rand_z
        self.target_vis(self.goal_pose)


    def uniform_exploration(self, action):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        #add wrist_3_joint with 0 to the action
        action_ = np.concatenate((action,0),axis=None)
        self.current_joints += action_
        action_sent = np.zeros(6)
        for i in range(5):
            if i == 3:
                self.current_joints[i] += np.pi
                if self.current_joints[i] > np.pi:
                    action_sent[i] = self.current_joints[i] % -np.pi
                elif self.current_joints[i] < -np.pi:
                    action_sent[i] = self.current_joints[i] % np.pi
                else:
                    action_sent[i] = self.current_joints[i]
                self.current_joints[i] -= np.pi
                action_sent[i] -= np.pi
            else:
                if self.current_joints[i] > np.pi:
                    action_sent[i] = self.current_joints[i] % -np.pi
                elif self.current_joints[i] < -np.pi:
                    action_sent[i] = self.current_joints[i] % np.pi
                else:
                    action_sent[i] = self.current_joints[i]

        goal.trajectory.points = [JointTrajectoryPoint(positions=action_sent, velocities=[0]*6, 
                                                                    time_from_start=rospy.Duration(self.duration))]
        self.client.send_goal(goal)
        self.client.wait_for_result()

        #get position of end effector
        position, rpy = self.get_pos(link_name='ee_link')
        #get low dimension state
        state = self.get_state(action,position)
        reward, terminal = self.get_reward(position,rpy,action)

        return state, action, reward, terminal


if __name__ == '__main__':
    arm = Ur5()
