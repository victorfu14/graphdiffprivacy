from turtle import color
from wsgiref.simple_server import demo_app
from typing import Literal
import numpy as np
import math
import argparse
import logging
import os
import matplotlib.pyplot as plt
# from undirected_graph_seq import *
# from directed_graph_seq import *
from utils import *

logger = logging.getLogger(__name__)
np.random.seed(0)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-1', type=str, default='sx-mathoverflow-a2q.txt')
    parser.add_argument('--dataset-2', type=str, default='sx-mathoverflow-a2q-adj.txt')
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--psum-size', type=int, default=10)
    parser.add_argument('--partition', type=str, default='naive', choices=['naive', 'random', 'binary'])
    parser.add_argument('--func', type=str, default='tri_count', 
        choices=['all', 'inkstar_count', 'outkstar_count', 'edge_count', 'seq_tri_count', 'alt_tri_count'])
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--out-dir', type=str)

    return parser.parse_args()

class directedGraphSequence:
    def __init__(self, file, 
                    timesteps=100, 
                    psum_size=10, 
                    directed=False,
                    options={'seq_tri_count': True,
                            'alt_tri_count': True,
                            'inkstar_count': False,
                            'outkstar_count': False,
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
                    i = str(bin(idx+1))[2:][::-1].find('1')
                    self.psum_data[key][i] = delta[key] + np.sum(self.psum_data[key][:i])
                    for j in range(i):
                        self.psum_data[key][j] = 0
                        self.psum_noisy[key][j] = 0
                    self.psum_noisy[key][i] = self.psum_data[key][i] + np.random.laplace(scale = gs / self.epsilon)

                    self.stat_data[key][idx] = self.psum_data[key].sum()
                    self.stat_noisy[key][idx] = int(self.psum_noisy[key].sum())
                    # self.psum_data[key][idx] = delta[key]
                    # self.psum_noisy[key].append(calc_psum(self.psum_data[key], gs, self.epsilon, idx))
                    
                    # self.stat_data[key][idx] = self.psum_data[key].sum()
                    # self.stat_noisy[key][idx] = int(self.psum_noisy[key][idx].sum())

                    self.global_sensitivity_list[key][idx] = gs
                
    
    def global_sensitivity(self, function: Literal['inkstar_count', 'outkstar_count', 'edge_count', 'seq_tri_count', 'alt_tri_count']):
        if function == 'seq_tri_count':
            return min(self.max_indegree, self.max_outdegree)
        if function == 'alt_tri_count':
            return min(self.max_indegree, self.max_outdegree) + self.max_outdegree
        if function == 'edge_count':
            return 1
        if function == 'inkstar_count':
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

            if self.options['seq_tri_count'] == True:
                delta['seq_tri_count'] += len([vertex for vertex in self.graph[edge[0]]['out'] if vertex in self.graph[edge[1]]['in']])
            if self.options['alt_tri_count'] == True:
                delta['alt_tri_count'] += len([vertex for vertex in self.graph[edge[0]]['out'] if vertex in self.graph[edge[1]]['out']])
                delta['alt_tri_count'] += len([vertex for vertex in self.graph[edge[0]]['in'] if vertex in self.graph[edge[1]]['in']])
                delta['alt_tri_count'] += len([vertex for vertex in self.graph[edge[0]]['in'] if vertex in self.graph[edge[1]]['out']])
            if self.options['inkstar_count'] == True:
                delta['inkstar_count'] += math.comb(len(self.graph[edge[0]]['in']), self.kstar) - math.comb(len(self.graph[edge[0]]['in']) - 1, self.kstar) 
            if self.options['outkstar_count'] == True:
                delta['outkstar_count'] += math.comb(len(self.graph[edge[1]]['out']), self.kstar) - math.comb(len(self.graph[edge[1]]['out']) - 1, self.kstar) 
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
                    self.global_sensitivity_list[key] / self.epsilon * math.log(self.timesteps) * math.log(1 / 0.1), 
                    linewidth=2.0, 
                    label='theoretical_error_T',
                    color='red'
                )
                ax.axhline(2 * self.global_sensitivity(key) / self.epsilon * math.sqrt(self.psum_size) * math.log(1 / 0.1), color='orange')
                ax.axhline(self.global_sensitivity(key) / self.epsilon * math.log(self.timesteps) * math.log(1 / 0.1), color='red')
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

def main():
    args = get_args()

    args.out_dir += args.dataset_1 + '_' + args.dataset_2 + '_T=' + str(args.timesteps) + '_' + args.func + '_' 
    args.out_dir += '_n=' + str(args.psum_size) + '_partition=' + args.partition
    if args.directed:
        args.out_dir += '_directed'
        
    logfile = args.out_dir + '_output.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=args.out_dir + '_output.log')
    logger.info(args)

    options={'seq_tri_count': False,
            'alt_tri_count': False,
            'inkstar_count': False,
            'outkstar_count': False,
            'edge_count': False}

    options[args.func] = True

    x = directedGraphSequence(
        file=args.dataset_1, 
        timesteps=args.timesteps, 
        psum_size=args.psum_size, 
        options=options
    )

    if args.dataset_2 != 'None':
        y = directedGraphSequence(
            file=args.dataset_2, 
            timesteps=args.timesteps, 
            psum_size=args.psum_size
        )

    psum_func = update_psum_binary if args.partition == 'binary' else update_psum_naive
    logger.info('Dataset 1')
    x.stream(psum_func)
    x.summary(args.delta, out_dir=args.out_dir)

    if args.dataset_2 != 'None':
        logger.info('Dataset 2')
        y.stream()
    
if __name__ == '__main__':
    main()