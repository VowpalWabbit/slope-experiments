# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:37:15 2019

@author: pasrinat
"""

# -*- coding: utf-8 -*-


f = open('commands_list_test.txt', 'w')
for logging_model_name in ["NNPredictor", "Tree"]:
    for target_model_name in ["NNPredictor", "Tree"]:
        for soften in ["friendly", "adversarial"]:
            for kernel in ["boxcar", "epanechnikov"]:
                for loss in ["triangular", "parabolic"]:
                    for lip in [1, 3, 10]:
                        for samples in [100,1000,10000,100000]:
                            f.write("--logging_model_name "+str(logging_model_name)+" --target_model_name "+str(target_model_name)+" --soften "+str(soften)+" --kernel "+str(kernel)+" --loss "+str(loss)+" --lip "+str(lip)+" --samples "+str(samples)+"\n")
f.close()

f = open('commands_list_test.txt', 'a')
for logging_model_name in ["NNPredictor", "Tree"]:
    for target_model_name in ["NNPredictor", "Tree"]:
        for loss in ["triangular", "parabolic"]:
            for lip in [1, 3, 10]:
                for kernel in ["boxcar", "epanechnikov"]:
                    for samples in [100,1000,10000,100000]:
                        f.write("--logging_model_name "+str(logging_model_name)+" --target_model_name "+str(target_model_name)+" --loss "+str(loss)+" --lip "+str(lip)+" --samples "+str(samples)+" --kernel "+str(kernel)+"\n")
f.close()