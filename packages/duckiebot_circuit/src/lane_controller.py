"""
Modified from here: https://github.com/duckietown/dt-core/blob/6d8e99a5849737f86cab72b04fd2b449528226be/packages/lane_control/include/lane_controller/controller.py
"""
from PID import PID


class LaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative pose of the Duckiebot in the current lane/

    This is a dual parallel PID controller for the lateral and angular error

    Args:
        Kp_d:       Proportional term for lateral deviation
        Kp_theta    Proportional term for heading deviation  
        Ki_d        Integral term for lateral deviation
        Ki_theta    Integral term for heading deviation
        Kd_d        Derivative term for lateral deviation
        Kd_theta    Derivative term for heading deviation
        sample_time The sample time for the PID controllers
        d_bounds    output limits for lateral deviation
        theta_bounds output limits for heading deviation
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.d_pid_controller = PID(
            self.parameters['Kp_d'],
            self.parameters['Ki_d'],
            self.parameters['Kd_d'],
            sample_time=self.parameters['sample_time'],
            output_limits=self.parameters['d_bounds']
        )
        self.theta_pid_controller =  PID(
            self.parameters['Kp_theta'],
            self.parameters['Ki_theta'],
            self.parameters['Kd_theta'],
            sample_time=self.parameters['sample_time'],
            output_limits=self.parameters['theta_bounds']
        )
        self.v_bar = 0.5

    def reset_controller(self):
        """This will reset both PID controllers"""
        self.d_pid_controller.set_auto_mode(False)
        self.d_pid_controller.set_auto_mode(True)
        self.theta_pid_controller.set_auto_mode(False)
        self.theta_pid_controller.set_auto_mode(True)

    def disable_controller(self):
        """This will disable both PID controllers"""
        self.d_pid_controller.set_auto_mode(False)
        self.theta_pid_controller.set_auto_mode(False)

    def enable_controller(self):
        """This will enable both PID controllers"""
        self.d_pid_controller.set_auto_mode(True)
        self.theta_pid_controller.set_auto_mode(True)

    def compute_control_actions(self, d_err, phi_err, wheels_cmd_exec):
        """Main Function, computes the control action given the current errors"""
        d_correct = self.d_pid_controller(d_err)
        phi_correct = self.theta_pid_controller(phi_err)

        omega = d_correct + phi_correct
        v = self.v_bar

        return v, omega
    
