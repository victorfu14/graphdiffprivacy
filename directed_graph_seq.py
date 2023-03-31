import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Literal
import logging

from utils import *

class directedGraphSequence:
    def __init__(self, file, 
                    timesteps=100, 
                    psum_size=10, 
                    directed=False,
                    options={'tri_count': True,
                            'kstar_count': False,
                            'edge_count': False}
                ):
        with open(file) as f:
            self.lines = f.readlines()
        f.close()
        self.sequence = partition(self.lines, timesteps)
        self.graph = {}
        self.graph_sequence = []
        self.psum_data = {}
        self.psum_noisy = {}
        self.stat_data = {}
        self.stat_noisy = {}
        self.global_sensitivity_list = {}
        self.timesteps = timesteps
        self.psum_size = psum_size
        self.options = options
        self.directed = directed
        for (key, value) in options.items():
            if value == True:
                self.psum_noisy[key] = []
                self.psum_data[key] = np.zeros(self.timesteps)
                self.stat_data[key] = np.zeros(self.timesteps)
                self.stat_noisy[key] = np.zeros(self.timesteps)
                self.global_sensitivity_list[key] = np.zeros(self.timesteps)


    def stream(self, calc_psum=update_psum_naive):
        self.max_indegree = 0
        self.max_outdegree = 0
        self.kstar = 7630
        self.epsilon = 1
        self.max_indegree_list = []
        self.max_outdegree_list = []

        # initiate graph sequences using (arXiv:2106.14756v1) definition
        for idx, lines in enumerate(self.sequence):
            incr_ver, decr_ver, incr_edge, decr_edge = [], [], [], []
            for line in lines:
                edge = line.split()
                if self.graph.get(edge[0]) == None:
                    if edge[0] not in incr_ver:
                        incr_ver.append(edge[0])
                if self.graph.get(edge[1]) == None:
                    if edge[1] not in incr_ver:
                        incr_ver.append(edge[1])
                incr_edge.append((edge[0], edge[1]))

            self.graph_sequence.append((incr_ver, decr_ver, incr_edge, decr_edge))

            delta = self.update(incr_ver, decr_ver, incr_edge, decr_edge)
            for (key, value) in self.options.items():
                if value == True:
                    gs = self.global_sensitivity(key)
                    self.psum_data[key][idx] = delta[key]
                    self.psum_noisy[key].append(calc_psum(self.psum_data[key], gs, self.epsilon, self.psum_size))
                    
                    self.stat_data[key][idx] = self.psum_data[key].sum()
                    self.stat_noisy[key][idx] = int(self.psum_noisy[key][idx].sum())

                    self.global_sensitivity_list[key][idx] = gs
                

    
    def global_sensitivity(self, function: Literal['tri_count', 'edge_count', 'kstar_count']):
        if function == 'tri_count':
            return min(self.max_indegree, self.max_outdegree)
        if function == 'edge_count':
            return 1
        if function == 'kstar_count':
            return (math.comb(self.max_indegree, self.kstar) - math.comb(self.max_indegree - 1, self.kstar))
    
    def update(self, incr_ver, decr_ver, incr_edge, decr_edge):
        delta = {}
        for ver in incr_ver:
            self.graph[ver] = {'in': [], 'out': []}
        
        for (key, value) in self.options.items():
            if value == True:
                delta[key] = 0

        for edge in incr_edge:
            self.graph[edge[0]]['in'].append(edge[1])
            self.graph[edge[1]]['out'].append(edge[0])
            
            self.max_indegree = max(len(self.graph[edge[0]]['in']), self.max_indegree)
            self.max_outdegree = max(len(self.graph[edge[1]]['out']), self.max_outdegree)
            if self.options['tri_count'] == True:
                delta['tri_count'] += len([vertex for vertex in self.graph[edge[0]]['out'] if vertex in self.graph[edge[1]]]['in'])
            if self.options['kstar_count'] == True:
                delta['kstar_count'] += math.comb(len(self.graph[edge[0]]['in']), self.kstar) - math.comb(len(self.graph[edge[0]]['in']) - 1, self.kstar) 
                # delta['kstar_count'] += math.comb(len(self.graph[edge[1]]), self.kstar) - math.comb(len(self.graph[edge[1]]) - 1, self.kstar)
            if self.options['edge_count'] == True:
                delta['edge_count'] += 1
            
        self.max_indegree_list.append(self.max_indegree)
        self.max_outdegree_list.append(self.max_outdegree)
        # print(self.max_degree)
        
        return delta
    
    def plot(self, plot=None, option: Literal['value', 'error'] = 'error', out_dir=None):
        plt_options = ['value', 'error']
        for (key, value) in self.options.items():
            if value == True:
                if plot == None:
                    fig, ax = plt.subplots()
                else:
                    fig, ax = plot
                ax.plot(np.abs(self.error[key]), linewidth=2.0, label='empricial error')
                ax.plot(
                    self.threshold[key],
                    linewidth=2.0, 
                    label='theoretical_error_y',
                    color='orange'
                )
                ax.plot(
                    np.minimum(self.max_degree_list) / self.epsilon * math.log(self.timesteps) * math.log(1 / 0.1), 
                    linewidth=2.0, 
                    label='theoretical_error_T',
                    color='red'
                )
                ax.axhline(2 * min(self.max_indegree, self.max_outdegree) / self.epsilon * math.sqrt(self.psum_size) * math.log(1 / 0.1), color='orange')
                ax.axhline(2 * min(self.max_indegree, self.max_outdegree) / self.epsilon * math.log(self.timesteps) * math.log(1 / 0.1), color='red')
                title = 'empricial_vs_theoretical_' + 'error'
                ax.set_ylabel(key)
                ax.set_xlabel('timestep')
                ax.set_title(title)
                ax.legend()
                ax.figure.savefig(out_dir + '_' + title + '.png')
            
        return fig, ax
    
    def summary(self, delta, out_dir):
        logger.info('Max indegree: {}'.format(self.max_indegree))
        logger.info('Max outdegree: {}'.format(self.max_outdegree))
        self.error = {}
        self.threshold = {}
        for (key, value) in self.options.items():
            if value == True:
                logger.info('{} real statistic: {}'.format(key, self.stat_data[key]))
                logger.info('{} private statistic: {}'.format(key, self.stat_noisy[key]))
                
                self.error[key] = np.abs(self.stat_noisy[key] - self.stat_data[key])
                self.threshold[key] = 2 * self.global_sensitivity_list[key] / self.epsilon * math.sqrt(self.psum_size) * math.log(1 / delta)
                # print(self.global_sensitivity_list[key])
                # print(self.threshold[key])
                error_large = np.array([err for (err, bound) in zip(self.error[key], self.threshold[key]) if err > bound])
                
                logger.info('Error probablity: {}'.format(error_large.size / self.timesteps))
        
        self.plot(out_dir=out_dir)