from model.ml_models import prediction
from conf.conf import settings
import argparse


parser = argparse.ArgumentParser(description="Taking parameters for models")
parser.add_argument(
    '--prediction_model_path',
    type=str,
    default=settings.rf_conf,
    help=f'Provide a path to the model (default: {settings.rf_conf})'
)
model_var = parser.parse_args()
path = model_var.prediction_model_path

prediction(settings.data_set, path)
