

import sys
# sys.path.append('C:\Users\29184\Documents\GitHub\EvolutionRL')
import AquaML
import os
# AquaML.config_logger('log.log')


# current_path = os.getcwd()

# files = os.listdir(current_path)

# default_ids = []

# for file in files:
#     if 'Default' in file:
#         AquaML.logger.info('detected Default in file name: ' + file)
#         Default_id = file[7:]
#         default_ids.append(eval(Default_id))
#     # AquaML.logger.info(file)

# if len(default_ids) == 0:
#     AquaML.logger.info('No Default folder detected, using Default0')
    
# else:
#     AquaML.logger.info('Default folder detected, number of Default folders: ' + str(len(default_ids)))
#     AquaML.logger.info('max Default folder id: ' + str(max(default_ids)))

AquaML.init(
    root_path='Test1',
    memory_path='Test1',
    wandb_project='Test1'
    
)
# AquaML.logger.error('test')
# AquaML.logger.critical('test')
# AquaML.logger.debug('test')