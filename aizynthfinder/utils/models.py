import numpy as np
import onnxruntime
import psutil


class LocalOnnxModel:

    def __init__(self, filename: str) -> None:
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = _get_thread_count_per_core()

        self.model = onnxruntime.InferenceSession(filename,
                                                  sess_options=session_options)
        self._model_inputs = self.model.get_inputs()
        self._model_output = self.model.get_outputs()[0]
        self._model_dimensions = int(self._model_inputs[0].shape[1])
        self.output_size = int(self._model_output.shape[1])

    def __len__(self) -> int:
        return self._model_dimensions

    def predict(self, *args: np.ndarray, **_: np.ndarray) -> np.ndarray:
        return self.model.run(
            [self._model_output.name],
            {
                model_input.name: input.astype(np.float32)
                for model_input, input in zip(self._model_inputs, list(args))
            },
        )[0]


def _get_thread_count_per_core() -> int:
    return psutil.cpu_count() // psutil.cpu_count(logical=False)
