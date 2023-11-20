import os
import csv
import pandas as pd
import numpy as np

class SigmaLossLogger():

    def __init__(self, t_bins):
        self.t_bins = t_bins
        self.t = []
        self.mean = []
        self.var = []

    def write_loss(self, t, loss_mean, loss_var):
        self.t.append(t)
        self.mean.append(loss_mean)
        self.var.append(loss_var)

    def reset_logger(self):
        self.t = []
        self.mean = []
        self.var = []

    def log_t_bins(self):
        t = np.array(self.t)
        mu = np.array(self.mean)
        # var = np.array(self.var)
        
        log_loss=[]
        for i in range(1,len(self.t_bins)):
            if i == len(self.t_bins)-1:
                mask = (t <= self.t_bins[i]) & (t > self.t_bins[i-1])
            else:
                mask = (t <= self.t_bins[i]) & (t > self.t_bins[i-1])

            mu_i = mu[mask].mean() if len(mu[mask]) else .0
            var_i = mu[mask].std() if len(mu[mask]) else .0
            #var_i=var[mask].mean()
            
            log_loss.append([(self.t_bins[i]+self.t_bins[i-1])/2, mu_i, var_i])
        return log_loss