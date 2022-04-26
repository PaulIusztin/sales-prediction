from abc import ABC


class Hook(ABC):
    def before_fit(self, runner):
        pass

    def after_fit(self, runner):
        pass
