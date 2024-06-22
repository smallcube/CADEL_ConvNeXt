import os
import argparse
import pprint
from run_networks_convnext_ensemble import model
import yaml



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/ImageNet_LT/resnet50_ensemble.yaml', type=str)
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()

    def update(config, args):
        # Change parameters
        config['local_rank'] = args.local_rank
        return config

    # ============================================================================
    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    config = update(config, args)

    training_opt = config['training_opt']
    if not os.path.isdir(training_opt['log_dir']):
        os.makedirs(training_opt['log_dir'])
    
    pprint.pprint(config)
    training_model = model(config)

    training_model.train()
            
    print('ALL COMPLETED.')
