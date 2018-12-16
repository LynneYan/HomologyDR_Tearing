import os
import sys
sys.path.append(os.path.abspath('../src'))
import file_util
import mapper_util
import homologyDR_util
import tda_util

def stage_1(_DATA_TYPE):
    print '\nSTAGE 1 Start.'
    file_util.create_folders(_DATA_TYPE)
    print 'STAGE 1 Complete.'


def stage_2(_DATA_TYPE):
    print '\nSTAGE 2 Start.'
    mapper_util._generate_Json_files_R(_DATA_TYPE)
    print 'STAGE 2 Complete.'
    

def stage_3(_DATA_TYPE):
    print '\nSTAGE 3 Start.'
    homologyDR_util._projection_R(_DATA_TYPE)
    print 'STAGE 3 Complete.'


def stage_4(_DATA_TYPE):
    print '\nSTAGE 4 Start.'
    tda_util._evaluation_R(_DATA_TYPE)
    tda_util._plot_evaluation_R(_DATA_TYPE)
    print 'STAGE 4 Complete.'

        
if __name__ == '__main__':
    _DATA_TYPE = ['octa', 'airfoil1']
    stage_1(_DATA_TYPE)
    stage_2(_DATA_TYPE)
    stage_3(_DATA_TYPE)
    stage_4(_DATA_TYPE)
