from ...utils import log_params, create_and_get_local_logger, SIGNATURE_SUBPART_SEPARATOR


class PreprocessingPipeline(list):
    def __init__(self, name, iterable=()):
        super().__init__(iterable)
        self.name = name
        self.logger = None

    def init(self, logger=None):
        self._set_logger(logger)
        self.logger.info(f'Initializing preprocessing pipeline: {repr(self)}')
        for preprocessing in self:
            preprocessing.init(logger=self.logger)
        return self

    def log_params(self, logger=None):
        self._set_logger(logger)
        params = {
            'preprocessing_pipeline_name': self.name,
            'preprocessing_pipeline_representation': repr(self),
            'preprocessing_pipeline': [p.name() for p in self],
        }
        log_params(params,
                   logger=self.logger)
        for preprocessing in self:
            preprocessing.log_params(logger=self.logger)
        return params

    def is_empty(self):
        return len(self) == 0

    def _set_logger(self, logger):
        if logger:
            self.logger = logger
        else:
            self.logger = self.logger if self.logger else create_and_get_local_logger(self.__class__.__name__)
        return self

    def assemble_signature(self):
        params_signatures = [p.assemble_signature() for p in self]
        params_signature = f'[{SIGNATURE_SUBPART_SEPARATOR.join(params_signatures)}]'
        signature = f'pipeline({params_signature})'
        return signature

    def __repr__(self):
        return f"<PreprocessingPipeline {self.name!r}: {list(self)}>"
