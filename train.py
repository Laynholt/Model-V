from config.config import Config
from pprint import pprint


config = Config.load_json('/workspace/ext_data/projects/model-v/config/templates/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing.json')
pprint(config, indent=4)

print('\n\n')
config = Config.load_json('/workspace/ext_data/projects/model-v/config/templates/predict/ModelV.json')
pprint(config, indent=4)

print('\n\n')
config = Config.load_json('/workspace/ext_data/projects/model-v/config/templates/predict/ModelV_1.json')
pprint(config, indent=4)