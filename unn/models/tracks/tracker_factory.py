from .kalman_filter_tracker import KalmanFilterTracker

class TrackerFactory:

    @classmethod
    def create(cls, cfg):

        method = cfg['method']
        if 'kwargs' not in cfg:
            cfg['kwargs'] = {}
        kwargs = cfg['kwargs']
        if method == 'kalman filter':
            tracker = KalmanFilterTracker(**kwargs)
        else:
            raise ValueError('Unrecognized Tracker Method: ' + method)
        return tracker

