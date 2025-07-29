
import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger
from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.layer_output_utils import LayerOutputUtil # MCW Program lines added. Generate Layer-wise intermediate outputs

AimetLogger.set_level_for_all_areas(logging.DEBUG)

model_path = "./model_image_bev_output.h5"
data_path = "data"
model = tf.keras.models.load_model(model_path)

# Number of Samples for QAT
NUM_SAMPLES = 10

sim = QuantizationSimModel(
    model,
    quant_scheme=QuantScheme.training_range_learning_with_tf_init,
    rounding_mode="nearest",
    default_output_bw=8,
    default_param_bw=8,
)

for layer in sim.model.layers:
    if not isinstance(layer, tf.keras.layers.InputLayer):  # Input Layers aren't wrapped
        if (
            "tf.math.cumsum" in layer._layer_to_wrap.name
        ):  # Name of the layer we want to update, layer -> QcQuantizeWrapper, layer_to_wrap -> original layer
            print(layer._layer_to_wrap.name)  # Printing to verify name

            print(layer.input_quantizers)  # See the num quantizers
            for q in layer.input_quantizers:  # Loop through all potential quantizers and change to 32
                q.bitwidth = 32

            print(layer.output_quantizers)
            for q in layer.output_quantizers:
                q.bitwidth = 32

            print(
                layer.param_quantizers, end="\n\n"
            )  # There should be no param quantizers, because there are no weights

            for q in layer.input_quantizers:
                print(q.bitwidth)
            for q in layer.output_quantizers:
                print(q.bitwidth)
        
        if (
            "tf.compat.v1.gather_nd" in layer._layer_to_wrap.name
        ):  # Name of the layer we want to update, layer -> QcQuantizeWrapper, layer_to_wrap -> original layer
            print("\nAthi Gather: ", layer._layer_to_wrap.name)  # Printing to verify name

            print(layer.input_quantizers)  # See the num quantizers
            for q in layer.input_quantizers:  # Loop through all potential quantizers and change to 32
                q.bitwidth = 32

            print(layer.output_quantizers)
            for q in layer.output_quantizers:
                q.bitwidth = 32

            print(
                layer.param_quantizers, end="\n\n"
            )  # There should be no param quantizers, because there are no weights

            for q in layer.input_quantizers:
                print(q.bitwidth)
            for q in layer.output_quantizers:
                print(q.bitwidth)
        
        if (
            "tf.__operators__.add" in layer._layer_to_wrap.name
        ):  # Name of the layer we want to update, layer -> QcQuantizeWrapper, layer_to_wrap -> original layer
            print("\nAthi Add: ", layer._layer_to_wrap.name)  # Printing to verify name

            print(layer.input_quantizers)  # See the num quantizers
            for q in layer.input_quantizers:  # Loop through all potential quantizers and change to 32
                q.bitwidth = 32

            print(layer.output_quantizers)
            for q in layer.output_quantizers:
                q.bitwidth = 32

            print(
                layer.param_quantizers, end="\n\n"
            )  # There should be no param quantizers, because there are no weights

            for q in layer.input_quantizers:
                print(q.bitwidth)
            for q in layer.output_quantizers:
                print(q.bitwidth)

        if (
            "tf.math.subtract" in layer._layer_to_wrap.name
        ):  # Name of the layer we want to update, layer -> QcQuantizeWrapper, layer_to_wrap -> original layer
            print("\nAthi Subtract: ", layer._layer_to_wrap.name)  # Printing to verify name

            print(layer.input_quantizers)  # See the num quantizers
            for q in layer.input_quantizers:  # Loop through all potential quantizers and change to 32
                q.bitwidth = 32

            print(layer.output_quantizers)
            for q in layer.output_quantizers:
                q.bitwidth = 32

            print(
                layer.param_quantizers, end="\n\n"
            )  # There should be no param quantizers, because there are no weights

            for q in layer.input_quantizers:
                print(q.bitwidth)
            for q in layer.output_quantizers:
                print(q.bitwidth)

        if (
            "tf.math.truediv" in layer._layer_to_wrap.name
        ):  # Name of the layer we want to update, layer -> QcQuantizeWrapper, layer_to_wrap -> original layer
            print("\nAthi Div: ", layer._layer_to_wrap.name)  # Printing to verify name

            print(layer.input_quantizers)  # See the num quantizers
            for q in layer.input_quantizers:  # Loop through all potential quantizers and change to 32
                q.bitwidth = 32

            print(layer.output_quantizers)
            for q in layer.output_quantizers:
                q.bitwidth = 32

            print(
                layer.param_quantizers, end="\n\n"
            )  # There should be no param quantizers, because there are no weights

            for q in layer.input_quantizers:
                print(q.bitwidth)
            for q in layer.output_quantizers:
                print(q.bitwidth)       


def load_input(data_path: str, sample_number: int) -> Dict[str, np.ndarray]:
    image = np.load(os.path.join(data_path, f"camera_front_main_image_{sample_number}.npy"))
    radar = np.load(os.path.join(data_path, f"radar_bev_{sample_number}.npy"))
    return {"camera_front_main/image": image, "radar_bev": radar}


def pass_calibration_data(sim_model, data_path):
    for sample in range(NUM_SAMPLES):
        print(f"Sample {sample}")
        X = load_input(data_path, sample)
        sim_model(X)
    

sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=data_path)
sim.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

X = load_input(data_path, 0)
input_batch = (X["camera_front_main/image"], X["radar_bev"])

fp32_layer_output_util = LayerOutputUtil(model=model, save_dir='FP32_H5_layer_outputs')
fp32_layer_output_util.generate_layer_outputs(input_batch)

quantsim_layer_output_util = LayerOutputUtil(model=sim.model, save_dir='AIMET_layer_outputs')
quantsim_layer_output_util.generate_layer_outputs(input_batch)

