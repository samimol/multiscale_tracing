# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:30:58 2023

@author: Sami
"""

import argparse
parser = argparse.ArgumentParser(description="multiscale network")

parser.add_argument('--num_machines', type=int, default=23)
parser.add_argument('--one_scale', action='store_true')
