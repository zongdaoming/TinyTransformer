import numpy as np
from numpy.linalg import inv
from .base_tracker import BaseTracker, BaseTrackRecord


def KalmanFilterUpdate(X, P, Y, H, R):
    '''
    Function:
        Predict position matrix and covariance matrix for next frame
        Update the estimation

    Input:
        X: np.array [4, 1] cx, vx, cy, vy
        P: np.array [4, 4] # covariance matrix
        Y: np.array [2, 1] cx, cy # detected result
        H: np.array [2, 4] # transfrom matrix
        R: np.array [4, 4] # error matrix for detected result

    Ouput:
        new_X: np.array [4, 1]
        new_P: np.array [2, 1]
    '''

    IM = H.dot(X)
    IS = R + H.dot(P).dot(H.transpose())
    K = P.dot(H.transpose()).dot(inv(IS))
    new_X = X + K.dot(Y - IM)
    new_P = P - K.dot(IS).dot(K.transpose())
    return new_X, new_P


def KalmanFilterPredict(X, P, A, Q):
    '''
    Function:
        Predict position matrix and covariance matrix for next frame

    Input:
        X: np.array [4, 1] cx, vx, cy, vy
        P: np.array [4, 4] # covariance matrix
        A: np.array [4, 4] # transform matrix for current X
        Q: np.array [4, 4] # error matrix for current P

    Ouput:
        new_X: np.array [4, 1]
        new_P: np.array [2, 1]
    '''

    new_X = np.dot(A, X)
    new_P = A.dot(P).dot(A.transpose()) + Q
    return new_X, new_P

def KalmanFilterLoop(X, P, H, R, Y, A, Q):
    '''
    Function:
        Predict position matrix and covariance matrix for next frame
        Update the estimation

    Input:
        X: np.array [4, 1] cx, vx, cy, vy
        P: np.array [4, 4] # covariance matrix
        H: np.array [2, 4] # transfrom matrix
        R: np.array [4, 4] # error matrix for detected result
        Y: np.array [2, 1] cx, cy # detected result
        A: np.array [4, 4] # transform matrix for current X
        Q: np.array [4, 4] # error matrix for current P

    Ouput:
        new_X: np.array [4, 1]
        new_P: np.array [2, 1]
    '''
    
    tmp_X, tmp_P = KalmanFilterPredict(X, P, A, Q)
    new_X, new_P = KalmanFilterUpdate(tmp_X, tmp_P, Y, H, R)
    return new_X, new_P


def KalmanFilterEstimation(X, P, Y, Y_valid):
    '''
    Function:
        Estimate position matrix X and covariance matrix P for current frame

    Input:
        X: np.array [4, 1] cx, vx, cy, vy
        P: np.array [4, 4] # covariance matrix
        Y: np.array [2, 1] cx, cy # current detected result
        Y_valid: bool # whether use detected result

    Ouput:
        new_X: np.array [4, 1]
        new_P: np.array [2, 1]
    '''

    # 1. Generate transform matrix for detected position
    H = np.zeros((2, 4))
    H[0][0] = 1  # identical transform for cx
    H[1][2] = 1  # identical transform for cy

    # 2. Generate covariance matrix for detected position
    R = np.zeros((2, 2))
    R[0][0] = R[1][1] = 0.1 # 0.1 * Identity Matrix

    # 3. Generate status transform matrix for current position
    A = np.zeros((4, 4))
    A[0][0] = A[1][1] = A[2][2] = A[3][3] = 1  # Identity Matrix
    A[0][1] = 1  # cx' = cx + vx
    A[2][3] = 1  # cy' = cy + vy

    # 4. Generate error matrix for prediction variance
    Q = np.zeros((4, 4))
    Q[0][0] = Q[1][1] = Q[2][2] = Q[3][3] = 1 # Identity Matrix
    Q[0][1] = Q[1][0] = Q[2][3] = Q[3][2] = 1
    Q = Q * 0.0025

    if Y_valid:
        return KalmanFilterLoop(X, P, H, R, Y, A, Q)
    else:
        return KalmanFilterPredict(X, P, A, Q)

class KalmanFilterTrackRecord(BaseTrackRecord):

    def __init__(self, track_id, roi, score, cur_frame):
        super(KalmanFilterTrackRecord, self).__init__(track_id, roi, score, cur_frame)
        
        X = np.zeros((4, 1))
        X[0][0] = self.obj.cx
        X[2][0] = self.obj.cy

        Y = np.zeros((2, 1))
        Y[0][0] = self.obj.cx
        Y[1][0] = self.obj.cy

        P = np.zeros((4, 4))
        P[0][0] = P[1][1] = P[2][2] = P[3][3] = 1
        P = P * 5

        self.cur_X, self.cur_P = KalmanFilterEstimation(X, P, Y, True)

class KalmanFilterTracker(BaseTracker):

    def status_update(self, trk):
        Y = np.zeros((2, 1))
        if trk.last_update == self.cur_frame:
            Y_valid = True
            Y[0][0] = trk.obj.cx
            Y[1][0] = trk.obj.cy
            trk.h = trk.obj.h
            trk.w = trk.obj.w
        else:
            Y_valid = False

        X = trk.cur_X
        P = trk.cur_P

        trk.cur_X, trk.cur_P = KalmanFilterEstimation(X, P, Y, Y_valid)
        trk.cx = trk.cur_X[0][0]
        trk.cy = trk.cur_X[2][0]
        
        return trk

    def generate_track(self, roi, track_id, track_score, cur_frame):
        return KalmanFilterTrackRecord(track_id, roi, track_score, cur_frame)
