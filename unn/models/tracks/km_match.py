import numpy as np
import logging

logger = logging.getLogger('global')

class KMMatch:

    def __init__(self, numx, numy, score_mat):
        self.n = numx
        self.m = numy
        self.g = score_mat
        self.inf = 100000
        self.lx = [-self.inf for _ in range(self.n)]
        self.lh = [0 for _ in range(self.n)]
        self.ly = [0 for _ in range(self.m)]
        self.rh = [0 for _ in range(self.m)]
        self.match = [-1 for _ in range(self.m)]
        self.slack = [0 for _ in range(self.m)]
        for i in range(self.n):
            for j in range(self.m):
                self.lx[i] = max(self.lx[i], self.g[i][j])
        for i in range(self.n):
            for j in range(self.m):
                self.slack[j] = self.inf

            while True:
                for j in range(self.n):
                    self.lh[j] = 0
                for j in range(self.m):
                    self.rh[j] = 0
                if self.findpath(i):
                    break
                else:
                    self.update()

    def findpath(self, x):
        self.lh[x] = 1
        for i in range(self.m):
            if abs(self.lx[x] + self.ly[i] - self.g[x][i]) < 0.001 and self.rh[i] == 0:
                self.rh[i] = 1
                if self.match[i] == -1 or self.findpath(self.match[i]):
                    self.match[i] = x
                    return True
            else:
                self.slack[i] = min(self.slack[i], self.lx[x] + self.ly[i] - self.g[x][i])
        return False

    def update(self):
        delta = self.inf
        for i in range(self.m):
            if self.rh[i] == 0:
                delta = min(delta, self.slack[i])
        for i in range(self.n):
            if self.lh[i] == 1:
                self.lx[i] -= delta
        for i in range(self.m):
            if self.rh[i] == 1:
                self.ly[i] += delta

