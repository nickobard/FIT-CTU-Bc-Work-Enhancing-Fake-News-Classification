from .transformations import Truncation


def truncate(pipelines, fraction):
    for pipeline in pipelines:
        pipeline.insert(0, Truncation(fraction))
