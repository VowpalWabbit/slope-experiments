import numpy as np
import argparse
import sys
import json
import os
# import azure.storage.blob as azureblob


def parse_args(myargs):
    parser=argparse.ArgumentParser(description='Continuous CB Simulations')
    parser.add_argument('--seed', type=int, default=577, metavar='N',
                        help='random seed (default: 577)')
    parser.add_argument('--start_iter', type=int, default=1,
                        help='Which replicate number to start at')
    parser.add_argument('--total_iter', type=int, default=5,
                        help='Number of iterations')
    parser.add_argument('--feat_dim', type=int, default=5,
                        help='Dimensionality of feature space')
    parser.add_argument('--act_dim', type=int, default=1,
                        help='Dimensionality of action space')
    parser.add_argument('--lip', type=float, default=10,
                        help='Lipschitz constant for losses')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples')
    parser.add_argument('--kernel', type=str, default=None,
                        help='kernel function to use for smoothing: boxcar or epanechnikov')
    parser.add_argument('--soften', type=str, default=None,
                        help='stochastic tranformation for policy: freindly, adversarial, Neutral or None')
    parser.add_argument('--loss', type=str, default='triangular',
                        help='specify loss function: triangular or parabolic')
    parser.add_argument('--logging_model_name', type=str, default=None,
                        help='specify model type for logging policy: NNPredictor or Tree or None')
    parser.add_argument('--target_model_name', type=str, default='NNPredictor',
                        help='specify model type for target policy: NNPredictor or Tree or None')
    parser.add_argument('--command_num', type=int, default=0,
                        help='command line number from commands_list.txt')
    parser.add_argument('--expt_name', type=str, default="slope-results",
                        help='results will be stored in a folder expt_name in azure storage')
    args=parser.parse_args(myargs)
    return(args)

def upload_to_blob(blob_client, container_name, file):
    print('Uploading file {} to container [{}]...'.format(file,container_name))
    blob_client.create_blob_from_path(container_name,str(file), os.path.basename(file))
    
if __name__=='__main__':
    
    #STORAGE_ACCOUNT_NAME = #your storage account name
    #STORAGE_ACCOUNT_KEY = # your storage account key
    
    
    Args = parse_args(sys.argv[1:])
    print(Args, flush=True)

    import random
    import torch
    np.random.seed(Args.seed)
    random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    

    import CCBEnv
    from Estimators import SmoothedEstimator
    from Slope import Slope
    import Params
    
    Env = CCBEnv.CCBSimulatedEnv(lip=Args.lip,feat_dim=Args.feat_dim,act_dim=Args.act_dim, target_model_name=Args.target_model_name,logging_model_name=Args.logging_model_name, loss_type=Args.loss, soften = Args.soften)
    Env.train_logger(10000)
    Env.train_target(100)
    ground_truth=Env.ground_truth(100000)
    print("ground truth: ", ground_truth)

    print("[Experiment] Bandwidths: %s" % (",".join([str(x) for x in Params.hs])), flush=True)
    for i in range(Args.start_iter, Args.start_iter+Args.total_iter):
        np.random.seed(Args.seed+37*i)
        random.seed(Args.seed+37*i)
        print("generating logging data")
        data = Env.gen_logging_data(Args.samples)
        mses = []
        mses_dict = {}
        for h in Params.hs:
            #print("\nh ", h)
            estimator=SmoothedEstimator(h, Args.soften, Args.kernel)
            estimate = estimator.estimate(Env.target, data)
            mses.append((estimate-ground_truth)**2)
            mses_dict[h] = mses[-1]
        estimator=Slope(params={'estimator': SmoothedEstimator, 'hyperparams': Params.hs, 'soften': Args.soften, 'kernel':Args.kernel})
        estimate = estimator.estimate(Env.target, data)
        mses.append((estimate-ground_truth)**2)
        mses_dict['slope'] = mses[-1]
        
        
        print({k: v for k, v in sorted(mses_dict.items(), key=lambda item: item[1])}, flush=True)
        f = open('./results/command_num=%s_replicate=%d.json' % (Args.command_num,i), 'w')
        results = {}
        for j in range(len(Params.hs)):
            results[Params.hs[j]] = mses[j]
        results['Slope'] = mses[-1]
        results['index'] = estimator.index
        results['command_num'] = Args.command_num
        results['replicate'] = i
        results['ground_truth'] = ground_truth
        results['logging_model'] = Args.logging_model_name
        results['target_model'] = Args.target_model_name
        results['soften'] = Args.soften
        results['kernel'] = Args.kernel
        results['lip'] = Args.lip
        results['samples'] = Args.samples
        results['loss'] = Args.loss
        f.write(json.dumps(results))
        f.close()
        
        #blob_client = azureblob.BlockBlobService(account_name=STORAGE_ACCOUNT_NAME,account_key=STORAGE_ACCOUNT_KEY)
    
        #upload_to_blob(blob_client, Args.expt_name, os.path.basename('command_num=%s_replicate=%d.json' % (Args.command_num,i)))

    
