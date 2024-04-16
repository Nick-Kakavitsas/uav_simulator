import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import math
import os
from matrix_utils import hat, vee, expm_SO3
import datetime

class fcns(object):
    def __init__(self):
        self.tmp = 1

    def plotting(self, tList, xyzList, xyzDotList, rpyList, pqrList, pqrDotList='skip', ctrlList='skip', matlabData='skip',
                 eiIData='skip', matlabFMData='skip', inputJ=np.zeros(3), figNamePrefix=''):
        # Check if there's an appropriate directory
        dirname = figNamePrefix + "figures"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(f"Directory '{dirname}' created successfully.")
        else:
            print(f"Directory '{dirname}' already exists.")

        # Check that the shape of J is as expected
        if inputJ.shape == (3, 3):
            inputJ = np.diag(inputJ)

        # Make a readme in the figure directory to say what the colors mean on the plots
        infoTxt = figNamePrefix + ' - blue lines in the plots \n matlab data - orange lines in the plots'
        readmeFilePath = dirname + '/readme.txt'
        with open(readmeFilePath, 'w') as file:
            file.write(infoTxt)

        # Separate out the motor speeds for plotting
        x = [row[0] for row in xyzList]
        y = [row[1] for row in xyzList]
        z = [row[2] for row in xyzList]
        u = [row[0] for row in xyzDotList]
        v = [row[1] for row in xyzDotList]
        w = [row[2] for row in xyzDotList]
        roll = [row[0] for row in rpyList]
        pitch = [row[1] for row in rpyList]
        yaw = [row[2] for row in rpyList]
        p = [row[0] for row in pqrList]
        q = [row[1] for row in pqrList]
        r = [row[2] for row in pqrList]
        if not isinstance(pqrDotList, str):
            pDot = [row[0] for row in pqrDotList]
            qDot = [row[1] for row in pqrDotList]
            rDot = [row[2] for row in pqrDotList]
        if not isinstance(ctrlList, str):
            w1 = [row[0] for row in ctrlList]
            w2 = [row[1] for row in ctrlList]
            w3 = [row[2] for row in ctrlList]
            w4 = [row[3] for row in ctrlList]
        if not isinstance(eiIData, str):
            eix = [row[0] for row in eiIData]
            eiy = [row[1] for row in eiIData]
            eiz = [row[2] for row in eiIData]
            eIx = [row[3] for row in eiIData]
            eIy = [row[4] for row in eiIData]
            eIz = [row[5] for row in eiIData]

        # Separate out the states from matlab for plotting
        if not isinstance(matlabData, str):
            # Unpack states
            mat_t = matlabData[:, 0] + 1  # +1 to account for the settling time during plotting
            mat_x = matlabData[:, 1]
            mat_y = matlabData[:, 2]
            mat_z = matlabData[:, 3]
            mat_u = matlabData[:, 4]
            mat_v = matlabData[:, 5]
            mat_w = matlabData[:, 6]
            mat_p = matlabData[:, 7]
            mat_q = matlabData[:, 8]
            mat_r = matlabData[:, 9]
            mat_R = matlabData[:, 10:19]
            mat_eix = matlabData[:, 19]
            mat_eiy = matlabData[:, 20]
            mat_eiz = matlabData[:, 21]
            mat_eIx = matlabData[:, 22]
            mat_eIy = matlabData[:, 23]
            mat_eIz = matlabData[:, 24]

            # Calculate pqrDot and M's
            mat_pDot = []
            mat_qDot = []
            mat_rDot = []
            for i in range(len(mat_p)):
                if i == 1:
                    mat_pDot.append(0)
                    mat_qDot.append(0)
                    mat_rDot.append(0)
                else:
                    dt = mat_t[i - 1] - mat_t[i]
                    mat_pDot.append((mat_p[i - 1] - mat_p[i]) / dt)
                    mat_qDot.append((mat_q[i - 1] - mat_q[i]) / dt)
                    mat_rDot.append((mat_r[i - 1] - mat_r[i]) / dt)

            mat_est_M1 = (np.array(mat_pDot) * inputJ[0]).tolist()
            mat_est_M2 = (np.array(mat_qDot) * inputJ[1]).tolist()
            mat_est_M3 = (np.array(mat_rDot) * inputJ[1]).tolist()

            # Convert R to euler angles
            mat_rpy = np.zeros([len(mat_p), 3])
            for i in range(len(mat_R)):
                tempR = (np.reshape(mat_R[i, :], (3, 3)))
                tempRPY = Rotation.from_matrix(tempR.T).as_euler('xyz', degrees=False)
                mat_rpy[i, :] = np.array(tempRPY)

            mat_roll = mat_rpy[:, 0]
            mat_pitch = mat_rpy[:, 1]
            mat_yaw = mat_rpy[:, 2]

            if not isinstance(matlabFMData, str):
                mat_fMt = matlabFMData[:, 0]
                mat_f = matlabFMData[:, 1]
                mat_M1 = matlabFMData[:, 2]
                mat_M2 = matlabFMData[:, 3]
                mat_M3 = matlabFMData[:, 4]

                # Plot the control over time
                fig9, axes9 = plt.subplots(1, 3)

                # M1
                axes9[0].plot(mat_t, mat_est_M1)
                axes9[0].plot(mat_fMt, mat_M1)
                axes9[0].set_xlabel('t (s)')
                axes9[0].set_ylabel('M1')

                # M2
                axes9[1].plot(mat_t, mat_est_M2)
                axes9[1].plot(mat_fMt, mat_M2)
                axes9[1].set_xlabel('t (s)')
                axes9[1].set_ylabel('M2')

                # M3
                axes9[2].plot(mat_t, mat_est_M3)
                axes9[2].plot(mat_fMt, mat_M3)
                axes9[2].set_xlabel('t (s)')
                axes9[2].set_ylabel('M3')
                fig9.tight_layout()
                fig9.savefig(dirname + '/MvsEstMPlot.png')

        '''Plot the xyz over time'''
        fig1, axes1 = plt.subplots(1, 3)

        # x
        axes1[0].plot(tList, x)
        if not isinstance(matlabData, str):
            axes1[0].plot(mat_t, mat_x)
        axes1[0].set_xlabel('t (s)')
        axes1[0].set_ylabel('x (m)')
        axes1[0].set_ylim(-10, 10)

        # y
        axes1[1].plot(tList, y)
        if not isinstance(matlabData, str):
            axes1[1].plot(mat_t, mat_y)
        axes1[1].set_xlabel('t (s)')
        axes1[1].set_ylabel('y (m)')
        axes1[1].set_ylim(-10, 10)

        # z
        axes1[2].plot(tList, z)
        if not isinstance(matlabData, str):
            axes1[2].plot(mat_t, mat_z)  ########## Flip sign here for gymfc
        axes1[2].set_xlabel('t (s)')
        axes1[2].set_ylabel('z (m)')
        axes1[2].set_ylim(-10, 10)
        fig1.tight_layout()
        fig1.savefig(dirname + '/xyz.png')

        '''Plot the xyzDot over time'''
        fig2, axes2 = plt.subplots(1, 3)

        # u
        axes2[0].plot(tList, u)
        if not isinstance(matlabData, str):
            axes2[0].plot(mat_t, mat_u)
        axes2[0].set_xlabel('t (s)')
        axes2[0].set_ylabel('u (m/s)')
        axes2[0].set_ylim(-15, 15)

        # v
        axes2[1].plot(tList, v)
        if not isinstance(matlabData, str):
            axes2[1].plot(mat_t, mat_v)
        axes2[1].set_xlabel('t (s)')
        axes2[1].set_ylabel('v (m/s)')
        axes2[1].set_ylim(-15, 15)

        # w
        axes2[2].plot(tList, w)
        if not isinstance(matlabData, str):
            axes2[2].plot(mat_t, mat_w) ########## Flip sign here for gymfc
        axes2[2].set_xlabel('t (s)')
        axes2[2].set_ylabel('w (m/s)')
        fig2.tight_layout()
        fig2.savefig(dirname + '/xyzDot.png')
        axes2[1].set_ylim(-15, 15)

        '''Plot the rpy over time'''
        fig3, axes3 = plt.subplots(1, 3)

        # roll
        axes3[0].plot(tList, roll)
        if not isinstance(matlabData, str):
            axes3[0].plot(mat_t, mat_roll)
        axes3[0].set_xlabel('t (s)')
        axes3[0].set_ylabel('roll (rad)')
        axes3[0].set_ylim(-math.pi - 0.5, math.pi + 0.5)

        # pitch
        axes3[1].plot(tList, pitch)
        if not isinstance(matlabData, str):
            axes3[1].plot(mat_t, mat_pitch)
        axes3[1].set_xlabel('t (s)')
        axes3[1].set_ylabel('pitch (rad)')
        axes3[1].set_ylim(-math.pi - 0.5, math.pi + 0.5)

        # yaw
        axes3[2].plot(tList, yaw)
        if not isinstance(matlabData, str):
            axes3[2].plot(mat_t, mat_yaw)
        axes3[2].set_xlabel('t (s)')
        axes3[2].set_ylabel('yaw (rad)')
        axes3[2].set_ylim(-math.pi - 0.5, math.pi + 0.5)
        fig3.tight_layout()
        fig3.savefig(dirname + '/rpy.png')

        '''Plot the pqr over time'''
        fig4, axes4 = plt.subplots(1, 3)

        # p
        axes4[0].plot(tList, p)
        if not isinstance(matlabData, str):
            axes4[0].plot(mat_t, mat_p)
        axes4[0].set_xlabel('t (s)')
        axes4[0].set_ylabel('roll rate (rad/s)')

        # q
        axes4[1].plot(tList, q)
        if not isinstance(matlabData, str):
            axes4[1].plot(mat_t, mat_q)
        axes4[1].set_xlabel('t (s)')
        axes4[1].set_ylabel('pitch rate (rad/s)')

        # r
        axes4[2].plot(tList, r)
        if not isinstance(matlabData, str):
            axes4[2].plot(mat_t, mat_r)
        axes4[2].set_xlabel('t (s)')
        axes4[2].set_ylabel('yaw rate (rad/s)')
        fig4.tight_layout()
        fig4.savefig(dirname + '/pqr.png')

        '''Plot the pqrDot over time'''
        if not isinstance(pqrDotList, str):
            fig5, axes5 = plt.subplots(1, 3)

            # pDot
            axes5[0].plot(tList, pDot)
            if not isinstance(matlabData, str):
                axes5[0].plot(mat_t, mat_pDot)
            axes5[0].set_xlabel('t (s)')
            axes5[0].set_ylabel('roll accel (rad/s^2)')

            # qDot
            axes5[1].plot(tList, qDot)
            if not isinstance(matlabData, str):
                axes5[1].plot(mat_t, mat_qDot)
            axes5[1].set_xlabel('t (s)')
            axes5[1].set_ylabel('pitch accel (rad/s^2)')

            # rDot
            axes5[2].plot(tList, rDot)
            if not isinstance(matlabData, str):
                axes5[2].plot(mat_t, mat_rDot)
            axes5[2].set_xlabel('t (s)')
            axes5[2].set_ylabel('yaw accel (rad/s^2)')
            fig5.tight_layout()
            fig5.savefig(dirname + '/pqrDot.png')

        '''Plot the control over time'''
        if not isinstance(ctrlList, str):
            fig6, axes6 = plt.subplots(2, 2)

            # w1
            axes6[0, 0].plot(tList, w1)
            axes6[0, 0].set_xlabel('t (s)')
            axes6[0, 0].set_ylabel('motor 1 (rad/s)')

            # w2
            axes6[0, 1].plot(tList, w2)
            axes6[0, 1].set_xlabel('t (s)')
            axes6[0, 1].set_ylabel('motor 2 (rad/s)')

            # w3
            axes6[1, 0].plot(tList, w3)
            axes6[1, 0].set_xlabel('t (s)')
            axes6[1, 0].set_ylabel('motor 3 (rad/s)')

            # w4
            axes6[1, 1].plot(tList, w4)
            axes6[1, 1].set_xlabel('t (s)')
            axes6[1, 1].set_ylabel('motor 4 (rad/s)')
            fig6.tight_layout()
            fig6.savefig(dirname + '/ctrlPlot.png')

        if not isinstance(eiIData, str):
            '''Plot gymfc controller ei over time'''
            fig7, axes7 = plt.subplots(1, 3)

            # eix
            axes7[0].plot(tList, eix)
            if not isinstance(matlabData, str):
                axes7[0].plot(mat_t, mat_eix)
            axes7[0].set_xlabel('t (s)')
            axes7[0].set_ylabel('ei_x (?)')

            # eiy
            axes7[1].plot(tList, eiy)
            if not isinstance(matlabData, str):
                axes7[1].plot(mat_t, mat_eiy)
            axes7[1].set_xlabel('t (s)')
            axes7[1].set_ylabel('ei_y (?)')

            # eiz
            axes7[2].plot(tList, eiz)
            if not isinstance(matlabData, str):
                axes7[2].plot(mat_t, mat_eiz)
            axes7[2].set_xlabel('t (s)')
            axes7[2].set_ylabel('ei_z (?)')
            fig7.tight_layout()
            fig7.savefig(dirname + '/ei.png')

            '''Plot gymfc controller eI over time'''
            fig8, axes8 = plt.subplots(1, 3)

            # eIx
            axes8[0].plot(tList, eIx)
            if not isinstance(matlabData, str):
                axes8[0].plot(mat_t, mat_eix)
            axes8[0].set_xlabel('t (s)')
            axes8[0].set_ylabel('eI_x (?)')

            # eIy
            axes8[1].plot(tList, eIy)
            if not isinstance(matlabData, str):
                axes8[1].plot(mat_t, mat_eiy)
            axes8[1].set_xlabel('t (s)')
            axes8[1].set_ylabel('eI_y (?)')

            # eIz
            axes8[2].plot(tList, eIz)
            if not isinstance(matlabData, str):
                axes8[2].plot(mat_t, mat_eiz)
            axes8[2].set_xlabel('t (s)')
            axes8[2].set_ylabel('eI_z (?)')
            fig8.tight_layout()
            fig8.savefig(dirname + '/eI.png')

        # plt.show()

    def newPlotting(self,gymFCInput,matlabStateDataInput = 'skip', twinDataInput = 'skip', estDataInput = 'skip', inputJ=np.zeros(3),figNamePrefix = ''):
        # Check if there's an appropriate directory
        dirname = figNamePrefix + "figures"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print(f"Directory '{dirname}' created successfully.")
        else:
            print(f"Directory '{dirname}' already exists.")

        # Check that the shape of J is as expected
        if inputJ.shape == (3,3):
            inputJ = np.diag(inputJ)

        # Make a readme in the figure directory to say what the colors mean on the plots
        infoTxt = figNamePrefix + ' - blue lines in the plots \n matlab data - orange lines in the plots'
        readmeFilePath = dirname + '/readme.txt'
        with open(readmeFilePath, 'w') as file:
            file.write(infoTxt)

        # Initialize the plotting variables with 'skip' strings for plotting
        matXYZData = matXYZDotData = matXYZ2DotData = matrpyData = matpqrData = matpqrDotData = matEstMData = mateiData = mateIData = \
            twinXYZData = twinXYZDotData = twinXYZ2DotData = twinrpyData = twinpqrData = twineiData = twineIData = \
            estXYZData = estXYZDotData = estXYZ2DotData = estXrpyData = estXpqrData = 'skip'

        # Separate out the motor speeds for plotting
        tList, xyzList, xyzDotList, xyz2DotList, rpyList, pqrList, pqrDotList, eiIData, gymFCctrlList = gymFCInput
        x = [row[0] for row in xyzList]
        y = [row[1] for row in xyzList]
        z = [row[2] for row in xyzList]
        xDot = [row[0] for row in xyzDotList]
        yDot = [row[1] for row in xyzDotList]
        zDot = [row[2] for row in xyzDotList]
        x2Dot = [row[0] for row in xyz2DotList]
        y2Dot = [row[1] for row in xyz2DotList]
        z2Dot = [row[2] for row in xyz2DotList]
        roll = [row[0] for row in rpyList]
        pitch = [row[1] for row in rpyList]
        yaw = [row[2] for row in rpyList]
        p = [row[0] for row in pqrList]
        q = [row[1] for row in pqrList]
        r = [row[2] for row in pqrList]
        pDot = [row[0] for row in pqrDotList]
        qDot = [row[1] for row in pqrDotList]
        rDot = [row[2] for row in pqrDotList]
        est_M1 = (np.array(pDot)*inputJ[0]).tolist()
        est_M2 = (np.array(qDot)*inputJ[1]).tolist()
        est_M3 = (np.array(rDot)*inputJ[1]).tolist()
        w1 = [row[0] for row in gymFCctrlList]
        w2 = [row[1] for row in gymFCctrlList]
        w3 = [row[2] for row in gymFCctrlList]
        w4 = [row[3] for row in gymFCctrlList]
        eix = [row[0] for row in eiIData]
        eiy = [row[1] for row in eiIData]
        eiz = [row[2] for row in eiIData]
        eIx = [row[3] for row in eiIData]
        eIy = [row[4] for row in eiIData]
        eIz = [row[5] for row in eiIData]
        gymfcXYZData = (tList, x, y, z, 'gymfc')
        gymfcXYZDotData = (tList, xDot, yDot, zDot, 'gymfc')
        gymfcXYZ2DotData = (tList, x2Dot, y2Dot, z2Dot, 'gymfc')
        gymfcrpyData = (tList, roll, pitch, yaw, 'gymfc')
        gymfcpqrData = (tList, p, q, r, 'gymfc')
        gymfcpqrDotData = (tList, pDot, qDot, rDot, 'gymfc')
        gymfcEstMData = (tList, est_M1, est_M1, est_M3, 'gymfc')
        gymfceiData = (tList, eix, eiy, eiz, 'gymfc')
        gymfceIData = (tList, eIx, eIy, eIz, 'gymfc')

        # Separate out the motor speeds for plotting
        if not isinstance(twinDataInput, str):
            tList_twin, xyzList_twin, xyzDotList_twin, xyz2DotList_twin, rpyList_twin, pqrList_twin, eiIData_twin = twinDataInput
            x_twin = [row[0] for row in xyzList_twin]
            y_twin = [row[1] for row in xyzList_twin]
            z_twin = [row[2] for row in xyzList_twin]
            xDot_twin = [row[0] for row in xyzDotList_twin]
            yDot_twin = [row[1] for row in xyzDotList_twin]
            zDot_twin = [row[2] for row in xyzDotList_twin]
            x2Dot_twin = [row[0] for row in xyz2DotList_twin]
            y2Dot_twin = [row[1] for row in xyz2DotList_twin]
            z2Dot_twin = [row[2] for row in xyz2DotList_twin]
            roll_twin = [row[0] for row in rpyList_twin]
            pitch_twin = [row[1] for row in rpyList_twin]
            yaw_twin = [row[2] for row in rpyList_twin]
            p_twin = [row[0] for row in pqrList_twin]
            q_twin = [row[1] for row in pqrList_twin]
            r_twin = [row[2] for row in pqrList_twin]
            # pDot_twin = [row[0] for row in pqrDotList_twin]
            # qDot_twin = [row[1] for row in pqrDotList_twin]
            # rDot_twin = [row[2] for row in pqrDotList_twin]
            # est_M1_twin = (np.array(pDot_twin) * inputJ[0]).tolist()
            # est_M2_twin = (np.array(qDot_twin) * inputJ[1]).tolist()
            # est_M3_twin = (np.array(rDot_twin) * inputJ[1]).tolist()
            eix_twin = [row[0] for row in eiIData_twin]
            eiy_twin = [row[1] for row in eiIData_twin]
            eiz_twin = [row[2] for row in eiIData_twin]
            eIx_twin = [row[3] for row in eiIData_twin]
            eIy_twin = [row[4] for row in eiIData_twin]
            eIz_twin = [row[5] for row in eiIData_twin]

            # Set the tuples for the twin data
            twinXYZData = (tList_twin, x_twin, y_twin, z_twin, 'twin')
            twinXYZDotData = (tList_twin, xDot_twin, yDot_twin, zDot_twin, 'twin')
            twinXYZ2DotData = (tList_twin, x2Dot_twin, y2Dot_twin, z2Dot_twin, 'twin')
            twinrpyData = (tList_twin, roll_twin, pitch_twin, yaw_twin, 'twin')
            twinpqrData = (tList_twin, p_twin, q_twin, r_twin, 'twin')
            twineiData = (tList_twin, eix_twin, eiy_twin, eiz_twin,'twin')
            twineIData = (tList_twin, eIx_twin, eIy_twin, eIz_twin,'twin')

        # Separate out the states from matlab for plotting
        if not isinstance(matlabStateDataInput, str):
            # Unpack states
            mat_t = matlabStateDataInput[:, 0] + 1# +1 to account for the settling time during plotting
            mat_x = matlabStateDataInput[:, 1]
            mat_y = matlabStateDataInput[:, 2]
            mat_z = matlabStateDataInput[:, 3]
            mat_xDot = matlabStateDataInput[:, 4]
            mat_yDot = matlabStateDataInput[:, 5]
            mat_zDot = matlabStateDataInput[:, 6]
            mat_p = matlabStateDataInput[:, 7]
            mat_q = matlabStateDataInput[:, 8]
            mat_r = matlabStateDataInput[:, 9]
            mat_R = matlabStateDataInput[:, 10:19]
            mat_eix = matlabStateDataInput[:, 19]
            mat_eiy = matlabStateDataInput[:, 20]
            mat_eiz = matlabStateDataInput[:, 21]
            mat_eIx = matlabStateDataInput[:, 22]
            mat_eIy = matlabStateDataInput[:, 23]
            mat_eIz = matlabStateDataInput[:, 24]

            # Calculate xyz2Dot, pqrDot and M's for the matlab data
            mat_x2Dot = []
            mat_y2Dot = []
            mat_z2Dot = []
            mat_pDot = []
            mat_qDot = []
            mat_rDot = []
            for i in range(len(mat_p)):
                if i == 1:
                    mat_x2Dot.append(0)
                    mat_y2Dot.append(0)
                    mat_z2Dot.append(0)
                    mat_pDot.append(0)
                    mat_qDot.append(0)
                    mat_rDot.append(0)
                else:
                    dt = mat_t[i-1]-mat_t[i]
                    mat_x2Dot.append((mat_xDot[i-1]-mat_xDot[i])/dt)
                    mat_y2Dot.append((mat_yDot[i-1]-mat_yDot[i])/dt)
                    mat_z2Dot.append((mat_zDot[i-1]-mat_zDot[i])/dt)
                    mat_pDot.append((mat_p[i-1]-mat_p[i])/dt)
                    mat_qDot.append((mat_q[i-1]-mat_q[i])/dt)
                    mat_rDot.append((mat_r[i-1]-mat_r[i])/dt)

            mat_est_M1 = (np.array(mat_pDot)*inputJ[0]).tolist()
            mat_est_M2 = (np.array(mat_qDot)*inputJ[1]).tolist()
            mat_est_M3 = (np.array(mat_rDot)*inputJ[1]).tolist()

            # Convert R to euler angles
            mat_rpy = np.zeros([len(mat_p), 3])
            for i in range(len(mat_R)):
                tempR = (np.reshape(mat_R[i, :], (3, 3)))
                tempRPY = Rotation.from_matrix(tempR.T).as_euler('xyz', degrees=False)
                mat_rpy[i, :] = np.array(tempRPY)

            mat_roll = mat_rpy[:, 0]
            mat_pitch = mat_rpy[:, 1]
            mat_yaw = mat_rpy[:, 2]

            # Set the tuples for the matlab data
            matXYZData = (mat_t, mat_x, mat_y, mat_z, 'matlab')
            matXYZDotData = (mat_t, mat_xDot, mat_yDot, mat_zDot, 'matlab')
            matXYZ2DotData = (mat_t, mat_x2Dot, mat_y2Dot, mat_z2Dot, 'matlab')
            matrpyData = (mat_t, mat_roll, mat_pitch, mat_yaw, 'matlab')
            matpqrData = (mat_t, mat_p, mat_q, mat_r, 'matlab')
            matpqrDotData = (mat_t, mat_pDot, mat_qDot, mat_rDot, 'matlab')
            matEstMData = (mat_t, mat_est_M1, mat_est_M1, mat_est_M3, 'matlab')
            mateiData = (mat_t, mat_eix, mat_eiy, mat_eiz, 'matlab')
            mateIData = (mat_t, mat_eIx, mat_eIy, mat_eIz, 'matlab')

        # Separate out the states from the estimate for plotting
        if not isinstance(estDataInput, str):
            tList_est, xyzList_est, xyzDotList_est, xyz2DotList_est, rpyList_est, pqrList_est = estDataInput

            # Unpack states
            x_est = [row[0] for row in xyzList_est]
            y_est = [row[1] for row in xyzList_est]
            z_est = [row[2] for row in xyzList_est]
            xDot_est = [row[0] for row in xyzDotList_est]
            yDot_est = [row[1] for row in xyzDotList_est]
            zDot_est = [row[2] for row in xyzDotList_est]
            x2Dot_est = [row[0] for row in xyz2DotList_est]
            y2Dot_est = [row[1] for row in xyz2DotList_est]
            z2Dot_est = [row[2] for row in xyz2DotList_est]
            roll_est = [row[0] for row in rpyList_est]
            pitch_est = [row[1] for row in rpyList_est]
            yaw_est = [row[2] for row in rpyList_est]
            p_est = [row[0] for row in pqrList_est]
            q_est = [row[1] for row in pqrList_est]
            r_est = [row[2] for row in pqrList_est]

            # Set the tuples for the estimate data
            estXYZData = (tList_est, x_est, y_est, z_est, 'est')
            estXYZDotData = (tList_est, xDot_est, yDot_est, zDot_est, 'est')
            estXYZ2DotData = (tList_est, x2Dot_est, y2Dot_est, z2Dot_est, 'est')
            estXrpyData = (tList_est, roll_est, pitch_est, yaw_est, 'est')
            estXpqrData = (tList_est, p_est, q_est, r_est, 'est')

        '''Plot the xyz over time'''
        xyzYLabs = ('x (m)', 'y (m)', 'z (m)')
        xyzFigName = dirname + '/xyz.png'
        allXYZData = (gymfcXYZData, matXYZData, twinXYZData, estXYZData)
        self.plot1x3new(allXYZData,xyzYLabs,xyzFigName)


        '''Plot the xyzDot over time'''
        xyzDotYLabs = ('xDot (m/s)', 'yDot (m/s)', 'zDot (m/s)')
        xyzDotFigName = dirname + '/xyzDot.png'
        yLims = (-15, 15)
        allXYZDotData = (gymfcXYZDotData, matXYZDotData, twinXYZDotData, estXYZDotData)
        self.plot1x3new(allXYZDotData,xyzDotYLabs,xyzDotFigName, yLims)

        '''Plot the xyz2Dot over time'''
        xyz2DotYLabs = ('x2Dot (m/s^2)', 'y2Dot (m/s^2)', 'z2Dot (m/s^2)')
        xyz2DotFigName = dirname + '/xyz2Dot.png'
        allXYZ2DotData = (gymfcXYZ2DotData, matXYZ2DotData, twinXYZ2DotData, estXYZDotData)
        self.plot1x3new(allXYZ2DotData,xyz2DotYLabs,xyz2DotFigName)

        '''Plot the rpy over time'''
        rpyYLabs = ('roll (rad)', 'pitch (rad)', 'yaw (rad)')
        rpyFigName = dirname + '/rpy.png'
        yLims = (-math.pi-0.5, math.pi+0.5)
        allrpyData = (gymfcrpyData, matrpyData, twinrpyData, estXrpyData)
        self.plot1x3new(allrpyData,rpyYLabs,rpyFigName, yLims)

        '''Plot the pqr over time'''
        pqrYLabs = ('p (rad/s)', 'q (rad/s)', 'r (rad/s)')
        pqrFigName = dirname + '/pqr.png'
        allpqrData = (gymfcpqrData, matpqrData, twinpqrData, estXpqrData)
        self.plot1x3new(allpqrData,pqrYLabs,pqrFigName)

        '''Plot the pqrDot over time'''
        if not isinstance(pqrDotList,str):
            pqrDotYLabs = ('pDot (rad/s^2)', 'qDot (rad/s^2)', 'rDot (rad/s^2)')
            pqrDotFigName = dirname + '/pqrDot.png'
            yLims = (-math.pi - 0.5, math.pi + 0.5)
            allpqrDotData = (gymfcpqrDotData, matpqrDotData)
            self.plot1x3new(allpqrDotData,pqrDotYLabs,pqrDotFigName,yLims)

        '''Plot the moment estimates over time'''
        if not isinstance(pqrDotList, str):
            estMYLabs = ('M1 (Nm)', 'M2 (Nm)', 'M3 (Nm)')
            estMFigName = dirname + '/momentEst.png'
            yLims = (-10, 10)
            allEstMData = (gymfcEstMData, matEstMData)
            self.plot1x3new(allEstMData,estMYLabs,estMFigName,yLims)

        '''Plot the motor control over time'''
        if not isinstance(gymFCctrlList, str):
            motorLabs = ('motor 1 (rad/s)', 'motor 2 (rad/s)', 'motor 3 (rad/s)' , 'motor 4 (rad/s)')
            motorFigName = dirname + '/ctrlPlot.png'
            self.plot2x2(tList, w1, w2, w3, w4, motorLabs, motorFigName, 'skip', 'skip')

        if not isinstance(eiIData,str):
            '''Plot ei over time'''
            eiYLabs = ('ei_x (?)', 'ei_y (?)', 'ei_z (?)')
            eiFigName = dirname + '/ei.png'
            alleiData = (gymfceiData, mateiData, twineiData)
            self.plot1x3new(alleiData,eiYLabs,eiFigName)

            '''Plot eI over time'''
            eIYLabs = ('eI_x (?)', 'eI_y (?)', 'eI_z (?)')
            eIFigName = dirname + '/eI.png'
            alleIData = (gymfceIData, mateIData, twineIData)
            self.plot1x3new(alleIData,eIYLabs,eIFigName)

        # plt.show()

    def plot1x3new(self,allData,ylabs, figName, yLims = 'skip'):
        '''
        Input a tuple of data ('allData') contianing the following:
            * data set 1
                - time vector corresponding to data
                - x data 1
                - y data 1
                - z data 1
                - legend entry
            * data set 2
                - time vector corresponding to data
                - x data 2
                - y data 2
                - z data 2
                - legend entry
            * etc

        ylabs = tuple with ylabel1, ylabel2, ylabel3

        yLims = tuple with low and high value

        Can also input additional data and labels in t2/3, x2/3, y2/3, z2/3
        '''

        # Setup the figure with 3 subplots
        fig, axes = plt.subplots(3, 1)

        # Loop through data and plot each set on each axis
        for i, dataSet in enumerate(allData):
            if not dataSet == 'skip':
                # Unpack data for this set
                t, x, y, z, legend = dataSet

                # Set the x limits based on the first set of data
                if i == 0:
                    # Set xlims
                    x0 = min(t) - max(t) / 100
                    xf = max(t)

                # Plot the data
                if legend == 'skip':
                    axes[0].plot(t, x)
                    axes[1].plot(t, y)
                    axes[2].plot(t, z)
                else:
                    axes[0].plot(t, x)
                    axes[1].plot(t, y)
                    axes[2].plot(t, z, label=legend)

        # Set plot characteristics
        for i, yLabel in enumerate(ylabs):
            axes[i].set_ylabel(yLabel)
            axes[i].set_xlabel('t (s)')
            axes[i].set_xlim(x0,xf)
        # Show the legend on the third plot
        axes[2].legend()

        # Set tight layout
        fig.tight_layout()

        # Save plot
        fig.savefig(figName)

    def plot2x2(self,t1,a1,b1,c1,d1,ylabs, figName, legendLabs = 'skip',
                t2 = 'skip',a2 = 'skip', b2 = 'skip', c2 = 'skip', d2 = 'skip',
                t3 = 'skip',a3 = 'skip', b3 = 'skip', c3 = 'skip', d3 = 'skip'):
        '''
        Input t1, and the appropriate y variables x1, y1, z1, and their y labels ylabs.

        ylabs = tuple with ylabel1, ylabel2, ylabel3, ylabel4

        legendLabs = tuple with legend entries for the number of input data

        yLims = tuple with low and high value

        Can also input additional data and labels in t2/3, x2/3, y2/3, z2/3
        '''

        # Unpack the labels
        ylab1, ylab2, ylab3, ylab4 = ylabs

        # Unpack legend labels
        skipLegs = 0
        if legendLabs[-1] == 'skip':
            skipLegs = 1
        else:
            if t3 != 'skip':
                leg1, leg2, leg3 = legendLabs
            elif t2 != 'skip':
                leg1, leg2 = legendLabs
            else:
                leg1 = legendLabs

        # Set xlims
        x0 = min(t1) - max(t1)/100
        xf = max(t1)

        # Make the figure
        fig, axes = plt.subplots(2, 2)

        # a
        axes[0, 0].plot(t1, a1)
        if not isinstance(t2, str):
            axes[0, 0].plot(t2, a2)
        if not isinstance(t3, str):
            axes[0, 0].plot(t3, a3)
            # axes[0, 0].set_ylim(0, 100)  # Only used when plotting fM
        axes[0, 0].set_xlabel('t (s)')
        axes[0, 0].set_ylabel(ylab1)
        axes[0, 0].set_xlim(x0, xf)

        # b
        axes[0, 1].plot(t1, b1)
        if not isinstance(t2, str):
            axes[0, 1].plot(t2, b2)
        if not isinstance(t3, str):
            axes[0, 1].plot(t3, b3)
            axes[0, 1].set_ylim(-10, 10) # Only used when plotting fM
        axes[0, 1].set_xlabel('t (s)')
        axes[0, 1].set_ylabel(ylab2)
        axes[0, 1].set_xlim(x0, xf)

        # c
        axes[1, 0].plot(t1, c1)
        if not isinstance(t2, str):
            axes[1, 0].plot(t2, c2)
        if not isinstance(t3, str):
            axes[1, 0].plot(t3, c3)
            axes[1, 0].set_ylim(-10, 10) # Only used when plotting fM
        axes[1, 0].set_xlabel('t (s)')
        axes[1, 0].set_ylabel(ylab3)
        axes[1, 0].set_xlim(x0, xf)

        # d
        if skipLegs == 1:
            axes[1, 1].plot(t1, d1)
            if not isinstance(t2, str):
                axes[1, 1].plot(t2, d2)
            if not isinstance(t3, str):
                axes[1, 1].plot(t3, d3)
                axes[0, 0].set_ylim(-10, 10) # Only used when plotting fM
        else:
            axes[1, 1].plot(t1, d1,label = leg1)
            if not isinstance(t2, str):
                axes[1, 1].plot(t2, d2, label = leg2)
            if not isinstance(t3, str):
                axes[1, 1].plot(t3, d3, label = leg3)
                axes[1, 1].set_ylim(-10, 10)
            axes[1, 1].legend()
        axes[1, 1].set_xlabel('t (s)')
        axes[1, 1].set_ylabel(ylab4)
        axes[1, 1].set_xlim(x0, xf)
        fig.tight_layout()
        fig.savefig(figName)

    def trajPicker(self, t, trajChoice, desAlt, xyz0):
        '''
        NK: Function to get the desired trajectories, based on Lee's examples
        trajChoice values
            * 1 = stationary trajectory
            * 2 = line trajectory
            * 3 = lissajous trajectory
        '''

        if trajChoice == 1:
            # Calculate the desired position and its first 4 derivatives
            xd = np.array([xyz0[0], desAlt, -desAlt])
            xd_dot = np.array([0., 0., 0.])
            xd_2dot = np.array([0., 0., 0.])
            xd_3dot = np.array([0., 0., 0.])
            xd_4dot = np.array([0., 0., 0.])

            # Calculate the desired b1 axis orientation, and its derivatives
            b1d = np.array([1., 0., 0.])
            b1d_dot = np.array([0., 0., 0.])
            b1d_2dot = np.array([0., 0., 0.])
        elif trajChoice == 2:
            # Set the height of the line
            height = 1

            # Calculate the desired position and its first 4 derivatives
            xd = np.array([xyz0[0] * 0.5 * t, xyz0[1], xyz0[2] + height])
            xd_dot = np.array([xyz0[0] * 0.5, 0, 0])
            xd_2dot = np.array([0, 0, 0])
            xd_3dot = np.array([0, 0, 0])
            xd_4dot = np.array([0, 0, 0])

            # Calcualte the desired b1 axis orientation, and its derivatives
            b1d = np.array([1, 0, 0])
            b1d_dot = np.array([0, 0, 0])
            b1d_2dot = np.array([0, 0, 0])

        # Added orientation reference states
        Rd = np.eye(3)
        Wd = np.array([0., 0., 0.])
        Wd_dot = np.array([0., 0., 0.])

        # Output the trajectory
        dTraj = (xd, xd_dot, xd_2dot, xd_3dot, xd_4dot, b1d, b1d_dot, b1d_2dot, Rd, Wd, Wd_dot)
        # print(dTraj)
        # input("dTraj^^")
        return dTraj

    def fM2w(self, fM, wMin, wMax, A, cM):
        '''
        NK: Convert from provided force/moment to individual forces per motor (based on our allocation matrix), then convert to individual rad/s values
        '''

        # try:
        #     f = fM[0]
        #     M1 = fM[1][0].tolist()
        #     M2 = fM[1][1].tolist()
        #     M3 = fM[1][2].tolist()
        #     fM = np.array([[f],[M1],[M2],[M3]])
        # except:
        #     fM = fM

        # Convert to individual forces per motor
        f_i = np.linalg.inv(A) @ (np.array([1,1,1,1])*fM)

        # Saturate forces to positive values
        # fCmd = np.maximum(f_i, 0)
        fCmd = f_i

        # Convert to rad/s per motor, don't clip at all
        # wRad = abs(fCmd)*cM
        wRad = np.sign(fCmd) * np.sqrt(abs(fCmd) / cM)
        # wRad = np.clip(np.sign(fCmd) * np.sqrt(abs(fCmd) / cM),0, 2200)

        # Convert to rad/s per motor, clip to limits of motor
        # wRad = np.clip(np.sign(fCmd)*np.sqrt(abs(fCmd) / cM), wMin, wMax)

        # Clip from (-wMin to wMax for reverse rotation cases)
        # wRad = np.clip(np.sign(fCmd) * np.sqrt(np.abs(fCmd) / cM), -wMax, wMax)

        # Clip from 0 to maxW
        # wRad = np.clip(np.sign(fCmd)*np.sqrt(np.abs(fCmd)/self.cM),-self.wMax,self.wMin)

        # Clip from -maxW to maxW
        # wRad = np.clip(np.sign(fCmd)*np.sqrt(np.abs(fCmd)/self.cM),-self.wMax,self.wMax)

        # Try just flipping the sign of the rotational velocity commands
        # wRad = np.clip(np.sign(fCmd)*np.sqrt(np.abs(fCmd)/self.cM),-self.wMax,self.wMax)
        # wRad[wRad < 0] *= -1

        # Try abs
        # wRad = np.abs(np.clip(np.sign(fCmd)*np.sqrt(np.abs(fCmd)/self.cM),-self.wMax,self.wMax))

        # Don't clip after conversion - Causes it to basically explode
        # wRad = np.sign(fCmd)*np.sqrt(np.abs(fCmd)/self.cM)

        return wRad

    def hat(self, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        hat_x = np.array([[0., -x3, x2],
                         [x3, 0., -x1],
                         [-x2, x1, 0.]])
        return hat_x

    def vee(self, M):
        # return np.array([M[2, 1], M[0, 2], M[1, 0]])
        return np.array([-M[1, 2], M[0, 2], -M[0, 1]])

    def attitude_errors(self, R, Rd, W, Wd):
        eR = 0.5 * self.vee(Rd.T.dot(R) - R.T.dot(Rd))
        eW = W - R.T.dot(Rd.dot(Wd))
        return (eR, eW)

    def position_errors(self, x, xd, v, vd):
        ex = x - xd
        ev = v - vd
        return (ex, ev)

    def satDot(self, y, yDot, sigma):
        z = np.zeros(y.shape)
        for i in range(len(y)):
            if y[i] > sigma:
                z[i] = 0
            elif y[i] < -sigma:
                z[i] = 0
            else:
                z[i] = yDot[i]

        return z

    def derivUnitVector(self,q,q_dot,q_2dot):
        '''
        Function to calculate a unit vector and its derivatives
        Taken from https://github.com/fdcl-gwu/uav_geometric_control/tree/master/matlab/aux_functions

        [u, u_dot, u_ddot] = derivUnitVector(q, q_dot, q_ddot)

        Inputs:
        q = A vector
        q_dot = The derivative of q
        q_ddot = The second derivative of q

        Outputs:
        u = The unit vector form of q
        u_dot = The first derivative of u
        u_ddot = The second derivative of u
        '''

        nq = np.linalg.norm(q)
        u = q / nq
        u_dot = (q_dot / nq) - q * np.dot(q, q_dot) / (nq**3)

        u_2dot = (q_2dot / nq
                  - (q_dot / nq**3) * (2 * np.dot(q, q_dot))
                  - (q / nq**3) * (np.dot(q_dot, q_dot) + np.dot(q, q_2dot))
                  + 3 * (q / nq**5) * np.dot(q, q_dot)**2)

        return (u,u_dot,u_2dot)

    def s(self,angle):
        return np.sin(angle)

    def c(self, angle):
        return np.cos(angle)

class Estimator:
    """Estimates the states of the UAV.

    This uses the estimator defined in "Real-time Kinematics GPS Based
    Telemetry System for Airborne Measurements of Ship Air Wake", but without
    the bias estimation terms.
    DOI: 10.2514/6.2019-2377

    x (3x1 numpy array) current position of the UAV [m]
    x: (3x1 numpy array) current position of the UAV [m]
    v: (3x1 numpy array) current velocity of the UAV [m/s]
    a: (3x1 numpy array) current acceleration of the UAV [m/s^s]
    b_a: (float) accelerometer bias in e3 direction [m/s^2]
    R: (3x3 numpy array) current attitude of the UAV in SO(3)
    W: (3x1 numpy array) current angular velocity of the UAV [rad/s]

    Q: (7x7 numpy array) variances of w_k
    P: (10x10 numpy array) covariances of the states

    t0: (datetime object) time at epoch
    t: (float) current time since epoch [s]
    t_prev: (float) time since epoch in the previous loop [s]

    W_pre: (3x1 numpy array) angular velocity of the previous loop [rad/s]
    a_imu_pre: (3x1 numpy array) acceleration of the previous loop [m/s^2]
    R_pre: (3x3 numpy array) attitude in the previous loop in SO(3)
    b_a_pre: (3x1 numpy array) accelerometer bias in the previous loop [m/s^2]

    g: (float) gravitational acceleration [m/s^2]
    ge3: (3x1 numpy array) gravitational acceleration direction [m/s^2]

    R_bi: (3x3 numpy array) transformation from IMU frame to the body frame
    R_bi_T: (3x3 numpy array) transformation from IMU frame to the body frame

    e3 : (3x1 numpy array) direction of the e3 axis
    eye3: (3x3 numpy array) 3x3 identity matrix
    eye10: (10x10 numpy array) 10x10 identity matrix

    zero3: (3x3 numpy array) 3x3 zero matrix
    """

    def __init__(self):
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.a = np.zeros(3)
        self.b_a = 0.0
        self.R = np.eye(3)
        self.W = np.zeros(3)

        # Variances of w_k
        self.Q = np.diag([
            0.001, 0.001, 0.001,  # acceleration
            0.025, 0.025, 0.025,  # angular velocity
            0.0001  # acclerometer z bias
        ])

        # Initial covariances of x
        self.P = np.diag([
            1.0, 1.0, 1.0,  # position
            1.0, 1.0, 1.0,  # velocity
            0.01, 0.01, 0.01,  # attitude
            1.0  # accelerometer z bias
        ])

        self.t0 = datetime.datetime.now()
        self.t = 0.0
        self.t_pre = 0.0

        self.W_pre = np.zeros(3)
        self.a_imu_pre = np.zeros(3)
        self.R_pre = np.eye(3)
        self.b_a_pre = 0.0

        self.g = 9.81
        self.ge3 = np.array([0.0, 0.0, self.g])

        # Transformation from IMU frame to the body frame.
        self.R_bi = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        self.R_bi_T = self.R_bi.T

        self.e3 = np.array([0.0, 0.0, 1.0])
        self.eye3 = np.eye(3)
        self.eye10 = np.eye(10)

        self.zero3 = np.zeros((3, 3))

    def prediction(self, a_imu, W_imu):
        """Prediction step of the estimator.

        Args:
            a_imu: (3x1 numpy array) acceleration measured by the IMU [m/s^2]
            W_imu: (3x1 numpy array) angular rate measured by the IMU [rad/s]
        """

        h = self.get_dt()

        self.R_pre = np.copy(self.R)
        self.W_pre = np.copy(self.W)
        self.b_a_pre = self.b_a * 1.0

        self.W = self.R_bi.dot(W_imu)
        self.R = self.R.dot(expm_SO3(h / 2.0 * (self.W + self.W_pre)))

        # This assumes IMU provide acceleration without g
        self.a = self.R.dot(self.R_bi).dot(a_imu) + self.b_a * self.e3
        a_pre = self.R_pre.dot(self.R_bi).dot(self.a_imu_pre) \
                + self.b_a_pre * self.e3

        self.x = self.x + h * self.v + h ** 2 / 2.0 * a_pre
        self.v = self.v + h / 2.0 * (self.a + a_pre)

        # Calculate A(t_{k-1})
        A = np.zeros((10, 10))
        A[0:3, 3:6] = self.eye3
        A[3:6, 6:9] = - self.R_pre.dot(hat(self.R_bi.dot(self.a_imu_pre)))
        A[3:6, 9] = self.e3
        A[6:9, 6:9] = -hat(self.R_bi.dot(W_imu))

        # Calculate F(t_{k-1})
        F = np.zeros((10, 7))
        F[3:6, 0:3] = self.R_pre.dot(self.R_bi)
        F[6:9, 3:6] = self.R_bi
        F[9, 6] = 1.0

        # Calculate \Psi using A(t)
        psi = self.eye10 + h / 2.0 * A

        A = self.eye10 + h * A.dot(psi)
        F = h * psi.dot(F)

        self.P = A.dot(self.P).dot(A.T) + F.dot(self.Q).dot(F.T)

        self.a_imu_pre = a_imu

    def imu_correction(self, R_imu, V_R_imu):
        """IMU correction step of the estimator.

        Args:
            R_imu: (3x3 numpy array) attitude measured by the IMU in SO(3)
            V_R_imu: (3x3 numpy array) attitude measurement covariance
        """

        imu_R = self.R.T.dot(R_imu).dot(self.R_bi_T)
        del_z = 0.5 * vee(imu_R - imu_R.T)

        H = np.block([self.zero3, self.zero3, self.eye3, np.zeros((3, 1))])
        H_T = H.T

        G = self.R_bi
        G_T = G.T

        V = V_R_imu

        S = H.dot(self.P).dot(H_T) + G.dot(V).dot(G_T)
        K = self.P.dot(H_T).dot(np.linalg.inv(S))

        X = K.dot(del_z)

        eta = X[6:9]
        self.R = self.R.dot(expm_SO3(eta))

        I_KH = self.eye10 - K.dot(H)
        self.P = I_KH.dot(self.P).dot(I_KH.T) \
                 + K.dot(G).dot(V).dot(G_T).dot(K.T)

    def gps_correction(self, x_gps, v_gps, V_x_gps, V_v_gps):
        """GPS correction step of the estimator.

        Args:
            x_gps: (3x1 numpy array) position measured by the GPS [m]
            v_gps: (3x1 numpy array) velocity measured by the GPS [m]
            V_x_gps: (3x1 numpy array) position measurement covariance
            V_v_gps: (3x1 numpy array) velocity measurement covariance
        """

        del_z = np.hstack((x_gps - self.x, v_gps - self.v))

        H = np.block([
            [self.eye3, self.zero3, self.zero3, np.zeros((3, 1))],
            [self.zero3, self.eye3, self.zero3, np.zeros((3, 1))]
        ])
        H_T = H.T

        V = np.block([
            [V_x_gps, self.zero3],
            [self.zero3, V_v_gps]
        ])

        S = H.dot(self.P).dot(H_T) + V
        K = self.P.dot(H_T).dot(np.linalg.inv(S))

        X = K.dot(del_z)

        dx = X[0:3]
        dv = X[3:6]
        db_a = X[9]

        self.x = self.x + dx
        self.v = self.v + dv
        self.b_a = self.b_a + db_a

        I_KH = self.eye10 - K.dot(H)
        self.P = I_KH.dot(self.P).dot(I_KH.T) + K.dot(V).dot(K.T)

    def get_dt(self):
        """Get the time difference between two loops.

        Return:
            dt: (float) time difference between two loops
        """

        self.t_pre = self.t * 1.0
        t_now = datetime.datetime.now()
        self.t = (t_now - self.t0).total_seconds()

        return self.t - self.t_pre

    def get_states(self):
        """Return the current states of the estimator.

        Return:
            x: (3x1 numpy array) current position of the UAV [m]
            v: (3x1 numpy array) current velocity of the UAV [m/s]
            a: (3x1 numpy array) current acceleration of the UAV [m/s^s]
            R: (3x3 numpy array) current attitude of the UAV in SO(3)
            W: (3x1 numpy array) current angular velocity of the UAV [rad/s]
        """
        return (self.x, self.v, self.a, self.R, self.W)

class controllers(object):

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
        momentConstant = 0.0135 # ctf from Lee
        epsilon = np.array([1,1,-1,-1])
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
        self.kx = 10
        self.kv = 8
        self.ki = 10
        self.c1 = 1.5
        self.sigma = 10

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

    def update(self,t,xyz,xyzDot,rpy,pqr,desTraj,ei,eI):
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
        (xd, xd_dot, xd_2dot, xd_3dot, xd_4dot, b1d, b1d_dot, b1d_2dot, _, _, _) = desTraj

        eX = X - xd
        eV = V - xd_dot
        A = (- self.kx * eX
             - self.kv * eV
             - (self.m * self.g * self.e3)
             + (self.m * xd_2dot)
             - self.ki * np.clip(ei,-self.sigma,self.sigma))

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

        Rc = np.transpose(np.reshape(np.array([b1c, b2c, b3c]),(3,3)))
        Rc_dot = np.transpose(np.reshape(np.array([b1c_dot, b2c_dot, b3c_dot]),(3,3)))
        Rc_ddot = np.transpose(np.reshape(np.array([b1c_ddot, b2c_2dot, b3c_2dot]),(3,3)))

        Wc = self.cmFcns.vee(Rc.T @ Rc_dot)
        Wc_dot = self.cmFcns.vee(Rc.T @ Rc_ddot - np.linalg.matrix_power(self.cmFcns.hat(Wc),2)) # Square Wc (non-elementwise)
        # Wc_dot = self.cmFcns.vee(Rc.T @ Rc_ddot - self.cmFcns.hat(Wc) @ self.cmFcns.hat(Wc))  # Square Wc (non-elementwise)

        # W3 = np.dot(R * e3, Rc * Wc)
        # W3_dot = np.dot(R * e3, Rc * Wc_dot) + np.dot(R * self.cmFcns.hat(W) * e3, Rc * Wc)

        '''Use the non-decoupled yaw form of the integral version of the Lee geometric controller'''
        [M, eI_dot, eR, eW] = self.attitudeControl(R, pqr, eI, Rc, Wc, Wc_dot)
        eY = 0
        eWy = 0

        '''Convert f and M to rad/s'''
        # fM = np.array([-f, *(-1*M)])
        # fM = np.array([-f, *(self.A2B@M.flatten())])
        # fM = np.array([-f, *np.flip(M.flatten())])
        # fM = np.array([f, M.flatten()])
        fM = np.vstack((f, M)).flatten()

        # Convert to rad/s per motor, clip to limits of motor
        wRad = self.cmFcns.fM2w(fM, self.wMin, self.wMax, self.A, self.cM)

        # print("time = {} \n force = {} \n M1 = {} \n M2 = {} \n M3 = {}".format(t, f, M[0], M[1], M[2]))

        return (wRad.flatten(), f, M, ei_dot, eI_dot.flatten())

    def attitudeControl(self, R, W, eI, Rd, Wd, Wd_dot):
        eR = np.reshape(0.5 * self.cmFcns.vee(Rd.T @ R - R.T @ Rd),(3,1))
        eW = np.reshape(W - R.T @ Rd @ Wd,(3,1))

        kR = np.diag([self.kR, self.kR, self.ky])
        kW = np.diag([self.kW, self.kW, self.kwy])

        # Note: \ is a line split in python
        a = - np.reshape(kR @ eR,(3,1))
        b = - np.reshape(kW @ eW,(3,1))
        c = - np.reshape(self.kI * eI,(3,1))
        # d1 = R.T @ Rd @ Wd
        # d2 = self.J @ R.T @ Rd @ Wd
        # d = np.reshape(self.cmFcns.hat(d1) @ d2,(3,1))
        d = np.reshape(self.cmFcns.hat(R.T @ Rd @ Wd) @ self.J @ R.T @ Rd @ Wd,(3,1))
        e = np.reshape(self.J @ R.T @ Rd @ Wd_dot,(3,1))
        M = a + b + c + d + e

        if len(M) > 3:
            print("M too large")
            input("")

        eI_dot = eW + self.c2 * eR

        return (M,eI_dot,eR,eW)