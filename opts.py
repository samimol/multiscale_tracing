# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:30:58 2023

@author: Sami
"""

import argparse
parser = argparse.ArgumentParser(description="multiscale network")

parser.add_argument('--num_networks', type=int, default=10)
parser.add_argument('--total_length', type=int, default=8)
parser.add_argument('--num_scales', type=int, default=4)
parser.add_argument('--one_scale', action='store_true')
parser.add_argument('--full_training', action='store_true')
