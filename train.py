from config.config import Config
from pprint import pprint


config = Config.load_json('/workspace/ext_data/projects/model-v/config/jsons/train/ModelV_BCE_MSE_Loss_AdamW_CosineAnnealing.json')
pprint(config, indent=4)

print('\n\n')
config = Config.load_json('/workspace/ext_data/projects/model-v/config/jsons/predict/ModelV.json')
pprint(config, indent=4)

print('\n\n')
config = Config.load_json('/workspace/ext_data/projects/model-v/config/jsons/predict/ModelV_1.json')
pprint(config, indent=4)