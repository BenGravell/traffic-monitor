from typing import cast

import pandas as pd

from traffic.detector import Detector, ModelConfig, RuntimeConfig, DetectorFailure
from traffic.detector.detections import Detections
from traffic.type_defs import Frame


class PlaybackDetector(Detector):
    def __init__(self, runtime_config: RuntimeConfig, model_config: ModelConfig) -> None:
        self.runtime_config = runtime_config
        self.model_config = model_config
        self.run_id = runtime_config.run_id
        self.history_df = self.load_detections_history()

    def load_detections_history(self) -> pd.DataFrame:
        path = f"data/runs/{self.run_id}/detections.parquet"
        return pd.read_parquet(path)

    def __call__(self, frame: Frame, frame_number: int) -> Detections:
        try:
            return Detections.from_dataframe(cast(pd.DataFrame, self.history_df.loc[frame_number]))
        except Exception as exc:
            raise DetectorFailure from exc
