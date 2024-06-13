import tritonclient.grpc as grpcclient

from .predictor_base import OnnxPredictorBase


class OnnxPredictorTriton(OnnxPredictorBase):
    def __init__(self, model_folder, triton_grcpclient=None):
        super().__init__(model_folder)
        if triton_grcpclient is None:
            self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001",
                # verbose=FLAGS.verbose,
                # ssl=FLAGS.ssl,
                # root_certificates=FLAGS.root_certificates,
                # private_key=FLAGS.private_key,
                # certificate_chain=FLAGS.certificate_chain,
            )
        else:
            self.triton_client = triton_grcpclient

    def _run_encoder(self, encoder_input_dict):
        encoder_input_ids = encoder_input_dict["pixel_values"]
        triton_input = [
            grpcclient.InferInput("pixel_values", encoder_input_ids.shape, "FP32")
        ]
        triton_input[0].set_data_from_numpy(encoder_input_ids)
        results = self.triton_client.infer(
            model_name="encoder_onnx",
            inputs=triton_input,
            #outputs=outputs,
            #client_timeout=FLAGS.client_timeout,
            #headers={"test": "1"},
            #compression_algorithm=FLAGS.grpc_compression_algorithm,
        )
        out_encoder = results.as_numpy("4605")
        return out_encoder

    def _run_decoder(self, decoder_input_dict):
        input_ids = decoder_input_dict["input_ids"]
        out_encoder = decoder_input_dict["encoder_hidden_states"]
        triton_input = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT32"),
            grpcclient.InferInput("encoder_hidden_states", out_encoder.shape, "FP32"),
        ]
        triton_input[0].set_data_from_numpy(input_ids)
        triton_input[1].set_data_from_numpy(out_encoder)
        results = self.triton_client.infer(
            model_name="decoder_onnx",
            inputs=triton_input,
            #outputs=outputs,
            #client_timeout=FLAGS.client_timeout,
            #headers={"test": "1"},
            #compression_algorithm=FLAGS.grpc_compression_algorithm,
        )
        n_output = len(results.get_response(as_json=True)["outputs"])
        logits = results.as_numpy("logits")
        past_key_values = {'past_key_value_input_' + str(k): results.as_numpy('past_key_value_output_' + str(k)) for k in
            range(n_output-1)}
        return logits, past_key_values

    def _run_decoder_with_past(self, decoder_input_dict):
        triton_input = []
        for key, val in decoder_input_dict.items():
            input_type = "FP32"
            if key == "input_ids":
                input_type = "INT32"
            infer_input = grpcclient.InferInput(key, val.shape, input_type)
            infer_input.set_data_from_numpy(val)
            triton_input.append(infer_input)
        results = self.triton_client.infer(
            model_name="decoder_with_past_onnx",
            inputs=triton_input,
            #outputs=outputs,
            #client_timeout=FLAGS.client_timeout,
            #headers={"test": "1"},
            #compression_algorithm=FLAGS.grpc_compression_algorithm,
        )
        n_output = len(results.get_response(as_json=True)["outputs"])
        logits = results.as_numpy("logits")
        past_key_values = {'past_key_value_input_' + str(k): results.as_numpy('past_key_value_output_' + str(k)) for k in
            range(n_output-1)}
        return logits, past_key_values