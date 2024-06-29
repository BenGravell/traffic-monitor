from traffic.tracker import TrackerType, RuntimeConfig, Tracker
from traffic.tracker.centroid import CentroidTracker


DETECTOR_TYPE_MAP = {
    TrackerType.CENTROID: CentroidTracker,
}


class TrackerFactory:
    @classmethod
    def create(cls, config: RuntimeConfig) -> Tracker:
        detector_cls = DETECTOR_TYPE_MAP[config.tracker_type]
        return detector_cls(config)
