import os
import sys
sys.path.append(os.path.abspath('../src'))
import file_util
import mapper_util
import homologyDR_util
import tda_util


def stage_1(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    print ('\nSTAGE 1 Start.')
    for exp in range (1,exps+1):
        for noise in noise_:
            for outlier in outlier_:
                if 'Swiss_hole' in _DATA_TYPE:
                    file_util._generate_U_SH_data(n_samples, noise, outlier, exp)
                if 'Fishing_net' in _DATA_TYPE:
                    file_util._generate_UFN_data(n_samples*3, noise, outlier, exp)
    print ('STAGE 1 Complete.')


def stage_2(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    print ('\nSTAGE 2 Start.')
    mapper_util._generate_Json_files(n_samples, noise_, outlier_, exps, _DATA_TYPE)
    print ('STAGE 2 Complete.')
    

def stage_3(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    print ('\nSTAGE 3 Start.')
    homologyDR_util._projection(n_samples, noise_, outlier_, exps, _DATA_TYPE)
    print ('STAGE 3 Complete.')


def stage_4(n_samples, noise_, outlier_, exps, _DATA_TYPE):
    print ('\nSTAGE 4 Start.')
    tda_util._evaluation(n_samples, noise_, outlier_, exps, _DATA_TYPE)
    tda_util._plot_evaluation(n_samples, noise_, outlier_, exps, _DATA_TYPE)
    print ('STAGE 4 Complete.')

        
if __name__ == '__main__':
    _DATA_TYPE = ['Swiss_hole', 'Fishing_net']
    noise_ = [0.04]
    outlier_ = [0]
    n_samples = 2000
    exps = 1
    stage_1(n_samples, noise_, outlier_, exps, _DATA_TYPE)
    stage_2(n_samples, noise_, outlier_, exps, _DATA_TYPE)
    stage_3(n_samples, noise_, outlier_, exps, _DATA_TYPE)
    stage_4(n_samples, noise_, outlier_, exps, _DATA_TYPE)
