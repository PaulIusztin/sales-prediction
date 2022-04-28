from corecontrol import controllers, proxies

from hooks.base import Hook


class CoreControlHook(Hook):
    def before_run(self, runner):
        experiment_controller = controllers.ExperimentController()
        experiment_name = experiment_controller.get_name_from_last()
        runner.experiment = experiment_controller.create(data={"name": experiment_name})
        runner.logger = runner.experiment.get_logger()
        for model in runner.models:
            model.set_logger(runner.logger)

        runner.dvc_dataset = runner.experiment.get_dataset()
        runner.codebase = runner.experiment.get_codebase()
