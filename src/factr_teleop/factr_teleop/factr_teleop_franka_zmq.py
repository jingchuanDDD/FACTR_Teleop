# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import time
import numpy as np

import rclpy
from sensor_msgs.msg import JointState

from factr_teleop.factr_teleop import FACTRTeleop
from python_utils.zmq_messenger import ZMQPublisher, ZMQSubscriber
from python_utils.global_configs import franka_left_real_zmq_addresses, franka_right_real_zmq_addresses


def create_array_msg(data):
    msg = JointState()
    msg.position = list(map(float, data))
    return msg


class FACTRTeleopFrankaZMQ(FACTRTeleop):
    """
    This class implements the required communication methods required by FACTRTeleopFranka using
    ZMQ. Communication between the leader teleop arm and the follower Franka arm is established
    using ZMQ subscribers and publishers. In addition, this class re-publishes all ZMQ communication
    to ROS via ROS publishers, such as the current Franka joint states, the current Franka joint
    position target commands, etc. 

    This class also demonstrates an example implementation of force-feedback for the leader gripper.
    The follower gripper publishes its torque information via ROS publisher. This class subscribes
    to the torque and computes joint torque for the leader gripper to achieve force-feedback.
    """

    def __init__(self):
        super().__init__()
        self.gripper_feedback_gain = self.config["controller"]["gripper_feedback"]["gain"]
        self.gripper_torque_ema_beta = self.config["controller"]["gripper_feedback"]["ema_beta"]
        self.gripper_external_torque = 0.0

    def set_up_communication(self):
        if self.name == "left":
            zmq_addresses = franka_left_real_zmq_addresses
        elif self.name == "right":
            zmq_addresses = franka_right_real_zmq_addresses
        else:
            raise ValueError(f"Invalid robot name '{self.name}'. Expected 'left' or 'right'.")

        # ZMQ publisher used to send joint position commands to the Franka follower arm
        self.franka_cmd_pub = ZMQPublisher(zmq_addresses["joint_pos_cmd_pub"])
        # ZMQ subscriber used to get the current joint position and velocity of the Franka follower arm
        self.franka_joint_state_sub = ZMQSubscriber(zmq_addresses["joint_state_sub"])
        # ROS publisher for re-publishing the current joint states of the Franka follower arm
        self.obs_franka_state_pub = self.create_publisher(JointState, f'/franka/{self.name}/obs_franka_state', 10)
        # ROS publisher for re-publishing Franka and gripper commands
        self.cmd_franka_pos_pub = self.create_publisher(JointState, f'/factr_teleop/{self.name}/cmd_franka_pos', 10)
        self.cmd_gripper_pos_pub = self.create_publisher(JointState, f'/factr_teleop/{self.name}/cmd_gripper_pos', 10)

        if self.enable_torque_feedback:
            # ZMQ subscriber used to get the extenral joint torque from the Franka follower arm
            self.franka_torque_sub = ZMQSubscriber(zmq_addresses["joint_torque_sub"])
            while self.franka_torque_sub.message is None:
                self.get_logger().info(f"Has not received Franka {self.name}'s external joint torques")
                time.sleep(0.1)
            # ROS publisher for re-publishing Franka's external joint torque
            self.obs_franka_torque_pub = self.create_publisher(JointState, f'/franka/{self.name}/obs_franka_torque', 10)
        
        if self.enable_gripper_feedback:
            # ROS subscriber for getting the gripper's torque information
            self.obs_gripper_torque_sub = self.create_subscription(
                JointState, f'/gripper/{self.name}/obs_gripper_torque', 
                self._gripper_external_torque_callback, 
                1,
            )

    def _gripper_external_torque_callback(self, data):
        gripper_external_torque = data.position[0]
        self.gripper_external_torque = self.gripper_torque_ema_beta * self.gripper_external_torque + \
            (1-self.gripper_torque_ema_beta) * gripper_external_torque
        
    def get_leader_gripper_feedback(self):
        return self.gripper_external_torque
    
    def gripper_feedback(self, leader_gripper_pos, leader_gripper_vel, gripper_feedback):
        torque_gripper = -1.0*gripper_feedback / self.gripper_feedback_gain
        return torque_gripper
    
    def get_leader_arm_external_joint_torque(self):
        external_torque = self.franka_torque_sub.message
        # re-publishe Franka external joint torque to ROS
        self.obs_franka_torque_pub.publish(create_array_msg(external_torque))
        return external_torque

    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        # send leader arm position as joint position target to the follower Franka arm
        self.franka_cmd_pub.send_message(leader_arm_pos)
        # re-publish joint position target command to ROS via ROS publisher
        self.cmd_franka_pos_pub.publish(create_array_msg(leader_arm_pos))

        # send gripper command to follower gripper
        self.cmd_gripper_pos_pub.publish(create_array_msg([leader_gripper_pos]))

        # publish the current Franka follower arm joint state to ROS for behavior cloning
        # data collection purposes.
        franka_state = self.franka_joint_state_sub.message
        self.obs_franka_state_pub.publish(create_array_msg(franka_state))
        

def main(args=None):
    rclpy.init(args=args)
    factr_teleop_franka_zmq = FACTRTeleopFrankaZMQ()

    try:
        while rclpy.ok():
            rclpy.spin(factr_teleop_franka_zmq)
    except KeyboardInterrupt:
        factr_teleop_franka_zmq.get_logger().info("Keyboard interrupt received. Shutting down...")
        factr_teleop_franka_zmq.shut_down()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

