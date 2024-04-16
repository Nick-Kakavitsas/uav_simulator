from matrix_utils import hat, vee, deriv_unit_vector, saturate
from integral_utils import IntegralError, IntegralErrorVec3
from common import fcns

import datetime
import numpy as np
import pdb


class Control:
    # def __init__(self, xyz0, rpy0):
    def __init__(self):
        '''
        NK: Use this to initialize the co
        '''
        # Set the properties of the iris
        # self.m = 1.5350
        # self.m = 1.5
        self.m = 2
        # self.J = np.array([[0.030039211414440, 0.001238311416000, 0.001238311416000],
        #                    [0.001238311416000, 0.030581726602367, 0.001238311416000],
        #                    [0.001238311416000, 0.001238311416000, 0.057572990136937]])
        # modelJ = np.array([0.029125, 0.029125, 0.055225])
        modelJ = np.array([0.02, 0.02, 0.04])
        self.J = np.diag(modelJ)
        self.cM = 5.840000000000000e-06
        # self.cM = 8.004e-4
        self.g = 9.81
        self.e3 = np.array([0., 0., 1.])
        self.wMin = 10
        self.wMax = 1100
        # Set the allocation matrix
        # L1 = 0.13
        # L21 = 0.2
        # L22 = 0.22
        # Lee matlab parameters
        d = 0.169
        momentConstant = 0.0135  # ctf from Lee
        epsilon = np.array([1, 1, -1, -1])
        # momentConstant = 0.06
        # self.A = np.array([[1, 1, 1, 1], [-L21, L22, L21, -L22], [-L1, L1, -L1, L1],
        #                    [epsilon[0] * momentConstant, epsilon[1] * momentConstant, epsilon[2] * momentConstant, epsilon[3] * momentConstant]])
        # self.A = np.array([[1, 1, 1, 1],
        #                    [-L21, L22, L21, -L22],
        #                    [-L1, L1, -L1, L1],
        #                    [epsilon[0] * momentConstant, epsilon[1] * momentConstant, epsilon[2] * momentConstant, epsilon[3] * momentConstant]])
        self.A = np.array([[1, 1, 1, 1],
                           [-d, d, d, -d],
                           [d, -d, d, -d],
                           [epsilon[0] * momentConstant, epsilon[1] * momentConstant, epsilon[2] * momentConstant, epsilon[3] * momentConstant]])
        # self.A = np.array([[1, 1, 1, 1],
        #                    [epsilon[0] * momentConstant, epsilon[1] * momentConstant, epsilon[2] * momentConstant, epsilon[3] * momentConstant],
        #                    [-d, d, -d, d],
        #                    [-d, d, d, -d]])

        # Set IC
        # self.xyz0 = xyz0
        # self.rpy0 = rpy0

        '''Set the gains of the geometric controller'''
        # # Position
        # self.kx = 100
        # self.kv = 15
        # self.ki = 50
        # self.c1 = 1.5
        # self.sigma = 10
        #
        # # Attitude
        # self.kR = 1.5
        # self.kW = 0.35
        # self.kI = 10
        # self.c2 = 1.5
        #
        # # Yaw
        # self.ky = 0.8
        # self.kwy = 0.15
        # self.kyI = 2
        # self.c3 = 2

        '''Original controller gains'''
        # Position
        self.kx = 16
        self.kv = 12
        self.ki = 0.01
        self.c1 = 1
        self.sigma = 1.8

        # Attitude
        self.kR = 1.5
        self.kW = 0.35
        self.kI = 10
        self.c2 = 2

        # Yaw
        self.ky = 0.8
        self.kwy = 0.15
        self.kyI = 2
        self.c3 = 2

        '''Setup the object for common functions'''
        self.cmFcns = fcns()

        '''Required initialization'''
        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_pre = 0.0
        self.dt = 1e-9

        # Current state
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.R = np.identity(3)
        self.W = np.zeros(3)

        # Desired states
        self.xd = np.zeros(3)
        self.xd_dot = np.zeros(3)
        self.xd_2dot = np.zeros(3)
        self.xd_3dot = np.zeros(3)
        self.xd_4dot = np.zeros(3)

        self.b1d = np.zeros(3)
        self.b1d[0] = 1.0
        self.b1d_dot = np.zeros(3)
        self.b1d_2dot = np.zeros(3)

        self.Wd = np.zeros(3)
        self.Wd_dot = np.zeros(3)

        self.Rd = np.identity(3)

        self.b3d = np.zeros(3)
        self.b3d_dot = np.zeros(3)
        self.b3d_2dot = np.zeros(3)

        self.b1c = np.zeros(3)
        self.wc3 = 0.0
        self.wc3_dot = 0.0

        # Flag to enable/disable integral control
        self.use_integral = False

        self.e1 = np.zeros(3)
        self.e1[0] = 1.0
        self.e2 = np.zeros(3)
        self.e2[1] = 1.0
        self.e3 = np.zeros(3)
        self.e3[2] = 1.0

        self.fM = np.zeros((4, 1))  # Force-moment vector
        self.f_total = 0.0  # Calculated forces required by each moter

        # Integral errors
        self.eIX = IntegralErrorVec3()  # Position integral error
        self.eIR = IntegralErrorVec3()  # Attitude integral error
        self.eI1 = IntegralError()  # Attitude integral error for roll axis
        self.eI2 = IntegralError()  # Attitude integral error for pitch axis
        self.eIy = IntegralError()  # Attitude integral error for yaw axis
        self.eIX = IntegralError()  # Position integral error

        self.sat_sigma = 1.8

    def run(self, states, desired):
        """Run the controller to get the force-moments required to achieve the 
        the desired states from the current state.

        Args:
            state: (x, v, a, R, W) current states of the UAV
            desired: (xd, xd_dot, xd_2dot, xd_3dot, xd_4dot, b1d, b1d_dot,
                b1d_2dot, is_landed) desired states of the UAV

        Return:
            fM: (4x1 numpy array) force-moments vector
        """

        self.x, self.v, self.a, self.R, self.W = states
        self.xd, self.xd_dot, self.xd_2dot, self.xd_3dot, self.xd_4dot, \
            self.b1d, self.b1d_dot, self.b1d_2dot, is_landed = desired

        # If the UAV is landed, do not run the controller and produce zero
        # force-moments.
        if is_landed:
            return np.zeros(4)

        # NK: Define new tuple for desired trajectory compatible with our function
        desTraj = (self.xd, self.xd_dot, self.xd_2dot, self.xd_3dot, self.xd_4dot,
                   self.b1d, self.b1d_dot, self.b1d_2dot)

        # Set ei and eI to zero
        ei = eI = np.zeros(3)

        # NK: Update the control call to use our function
        self.update(self.x, self.v, self.R, self.W, desTraj, ei, eI)

        return self.fM

    def update(self, xyz, xyzDot, rpy, pqr, desTraj, ei, eI):
        '''
        NK: Update function to calculate errors, and desired control based on position_control.m file in matlab folder

        Added ctrlChoice, removed time in lieu of inserting the desired trajectory (desTraj) term which already accounts for t
        '''

        # Get the current orientation matrix, based on the euler angle input rpy (roll, pitch, yaw)
        # Rcorrection = np.array([[-1, 0, 0],[0,-1,0],[0,0,1]])
        # r = Rotation.from_euler('xyz', np.array(rpy), degrees=False)
        # R = r.as_matrix()
        # R = rpy.as_matrix()
        # R = Rotation.from_euler('xyz', np.array(rpy), degrees=False).as_matrix()
        # R = self.A2B@r.as_matrix()
        R = rpy

        # W,X,V
        W = pqr
        X = xyz
        V = xyzDot

        '''
        Geometric controller code based on matlab implementation
        Also see - https://github.com/fdcl-gwu/uav_geometric_control/tree/master/matlab/position_control.m
        '''
        # Get desired trajectory
        (xd, xd_dot, xd_2dot, xd_3dot, xd_4dot, b1d, b1d_dot, b1d_2dot) = desTraj

        eX = X - xd
        eV = V - xd_dot
        A = (- self.kx * eX
             - self.kv * eV
             - (self.m * self.g * self.e3)
             + (self.m * xd_2dot)
             - self.ki * np.clip(ei, -self.sigma, self.sigma))

        ei_dot = eV + self.c1 * eX
        b3 = R @ self.e3
        f = -np.dot(A, b3)
        ea = (self.g * self.e3
              - (f / self.m) * b3
              - xd_2dot)
        A_dot = (- self.kx * eV
                 - self.kv * ea
                 + self.m * xd_3dot
                 - self.ki * self.cmFcns.satDot(ei, ei_dot, self.sigma))

        ei_ddot = ea + self.c1 * eV
        b3_dot = R @ self.cmFcns.hat(W) @ self.e3
        f_dot = -np.dot(A_dot, b3) - np.dot(A, b3_dot)
        eb = - (f_dot / self.m) * b3 - (f / self.m) * b3_dot - xd_3dot
        A_ddot = (- self.kx * ea
                  - self.kv * eb
                  + self.m * xd_4dot
                  - self.ki * self.cmFcns.satDot(ei, ei_ddot, self.sigma))

        [b3c, b3c_dot, b3c_2dot] = self.cmFcns.derivUnitVector(-A, -A_dot, -A_ddot)

        A2 = -self.cmFcns.hat(b1d) @ b3c
        A2_dot = -self.cmFcns.hat(b1d_dot) @ b3c - self.cmFcns.hat(b1d) @ b3c_dot
        A2_ddot = (- self.cmFcns.hat(b1d_2dot) @ b3c
                   - 2 * self.cmFcns.hat(b1d_dot) @ b3c_dot
                   - self.cmFcns.hat(b1d) @ b3c_2dot)

        [b2c, b2c_dot, b2c_2dot] = self.cmFcns.derivUnitVector(A2, A2_dot, A2_ddot)

        b1c = self.cmFcns.hat(b2c) @ b3c
        b1c_dot = self.cmFcns.hat(b2c_dot) @ b3c + self.cmFcns.hat(b2c) @ b3c_dot
        b1c_ddot = (self.cmFcns.hat(b2c_2dot) @ b3c
                    + 2 * self.cmFcns.hat(b2c_dot) @ b3c_dot
                    + self.cmFcns.hat(b2c) @ b3c_2dot)

        Rc = np.transpose(np.reshape(np.array([b1c, b2c, b3c]), (3, 3)))
        Rc_dot = np.transpose(np.reshape(np.array([b1c_dot, b2c_dot, b3c_dot]), (3, 3)))
        Rc_ddot = np.transpose(np.reshape(np.array([b1c_ddot, b2c_2dot, b3c_2dot]), (3, 3)))

        Wc = self.cmFcns.vee(Rc.T @ Rc_dot)
        Wc_dot = self.cmFcns.vee(Rc.T @ Rc_ddot - np.linalg.matrix_power(self.cmFcns.hat(Wc), 2))  # Square Wc (non-elementwise)
        # Wc_dot = self.cmFcns.vee(Rc.T @ Rc_ddot - self.cmFcns.hat(Wc) @ self.cmFcns.hat(Wc))  # Square Wc (non-elementwise)

        # W3 = np.dot(R * e3, Rc * Wc)
        # W3_dot = np.dot(R * e3, Rc * Wc_dot) + np.dot(R * self.cmFcns.hat(W) * e3, Rc * Wc)

        '''Use the non-decoupled yaw form of the integral version of the Lee geometric controller'''
        [M, eI_dot, eR, eW] = self.attitudeControl(R, pqr, eI, Rc, Wc, Wc_dot)
        eY = 0
        eWy = 0

        '''Use the conversion from original control.py'''
        M = np.array([M[0], -M[1], -M[2]])
        self.fM[0] = f
        for i in range(3):
            self.fM[i + 1] = M[i]

        '''Added variable storing like in original control.py'''
        R_T = self.R.T
        Rc_T = Rc.T
        W = self.W
        hatW = self.cmFcns.hat(W)
        e3 = self.e3
        self.f_total = f
        self.Rd = Rc
        self.Wd = Wc
        self.Wd_dot = Wc_dot

        # Roll / pitch
        self.b3d = b3c
        self.b3d_dot = b3c_dot
        self.b3d_2dot = b3c_2dot

        # Yaw
        self.b1c = b1c
        self.wc3 = e3 @ (R_T @ Rc @ Wc)
        self.wc3_dot = e3 @ (R_T @ Rc @ Wc_dot) \
                       - e3 @ (hatW @ R_T @ Rc @ Wc)

    def attitudeControl(self, R, W, eI, Rd, Wd, Wd_dot):
        eR = np.reshape(0.5 * self.cmFcns.vee(Rd.T @ R - R.T @ Rd), (3, 1))
        eW = np.reshape(W - R.T @ Rd @ Wd, (3, 1))

        kR = np.diag([self.kR, self.kR, self.ky])
        kW = np.diag([self.kW, self.kW, self.kwy])

        # Note: \ is a line split in python
        a = - np.reshape(kR @ eR, (3, 1))
        b = - np.reshape(kW @ eW, (3, 1))
        c = - np.reshape(self.kI * eI, (3, 1))
        # d1 = R.T @ Rd @ Wd
        # d2 = self.J @ R.T @ Rd @ Wd
        # d = np.reshape(self.cmFcns.hat(d1) @ d2,(3,1))
        d = np.reshape(self.cmFcns.hat(R.T @ Rd @ Wd) @ self.J @ R.T @ Rd @ Wd, (3, 1))
        e = np.reshape(self.J @ R.T @ Rd @ Wd_dot, (3, 1))
        M = a + b + c + d + e

        if len(M) > 3:
            print("M too large")
            input("")

        eI_dot = eW + self.c2 * eR

        return (M, eI_dot, eR, eW)

    def set_integral_errors_to_zero(self):
        """Set all integrals to zero."""
        self.eIX.set_zero()
        self.eIR.set_zero()
        self.eI1.set_zero()
        self.eI2.set_zero()
        self.eIy.set_zero()
        self.eIX.set_zero()

    def update_current_time(self):
        """Update the current time since epoch."""
        self.t_pre = self.t

        t_now = datetime.datetime.now()
        self.t = (t_now - self.t0).total_seconds()


    def get_current_time(self):
        """Return the current time since epoch.
        
        Return:
            t: (float) time since epoch [s]
        """
        t_now = datetime.datetime.now()
        return (t_now - self.t0).total_seconds()
