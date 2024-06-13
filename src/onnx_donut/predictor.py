import onnxruntime
import os

from .predictor_base import OnnxPredictorBase


class OnnxPredictor(OnnxPredictorBase):
    def __init__(self, model_folder, sess_options=None, providers=None):
        super().__init__(model_folder)
        if sess_options is None:
            # Avoid increase of memory usage between inferences
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_mem_pattern = False
        self.encoder = onnxruntime.InferenceSession(os.path.join(model_folder, 'encoder.onnx'), sess_options,
                                                    providers=providers)

        self.decoder = onnxruntime.InferenceSession(os.path.join(model_folder, 'decoder.onnx'), sess_options,
                                                    providers=providers)

        self.decoder_with_past = onnxruntime.InferenceSession(os.path.join(model_folder, 'decoder_with_past.onnx'),
                                                              sess_options, providers=providers)

    def _run_encoder(self, encoder_input_ids):
        return self.encoder.run(None, encoder_input_ids)[0]

    def _run_decoder(self, decoder_input_dict):
        out_decoder = self.decoder.run(None, decoder_input_dict)
        logits = out_decoder[0]
        past_key_values = {'past_key_value_input_' + str(k): out_decoder[k + 1] for k in
                            range(len(out_decoder[1:]))}
        return logits, past_key_values

    def _run_decoder_with_past(self, decoder_input_dict):
        out_decoder = self.decoder_with_past.run(None, decoder_input_dict)
        logits = out_decoder[0]
        past_key_values = {'past_key_value_input_' + str(i): pkv for i, pkv in enumerate(out_decoder[1:])}
        return logits, past_key_values

