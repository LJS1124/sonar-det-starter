"""sonar-det-starter：声呐图像目标检测入门工具包。"""
from .uatd import Annotation, Box, UATD_CLASSES, collect_classes, find_pairs, parse_xml

__all__ = ["Annotation", "Box", "UATD_CLASSES", "collect_classes", "find_pairs", "parse_xml"]
