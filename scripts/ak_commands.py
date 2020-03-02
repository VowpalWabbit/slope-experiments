i = 0
f = open('commands.sh', 'w')
for logging_model_name in ["NNPredictor", "Tree"]:
    for target_model_name in ["NNPredictor", "Tree"]:
        for soften in ["friendly", "adversarial"]:
            for kernel in ["boxcar"]:
                for loss in ["triangular"]:
                    for lip in [0.1, 1, 10]:
                        for samples in [10,100,1000]:
                            f.write("python3 ./src/Experiment.py --logging_model_name "+str(logging_model_name)+" --target_model_name "+str(target_model_name)+" --soften "+str(soften)+" --kernel "+str(kernel)+" --loss "+str(loss)+" --lip "+str(lip)+" --samples "+str(samples)+" --command_num "+str(i)+"\n")
                            i+=1
f.close()
