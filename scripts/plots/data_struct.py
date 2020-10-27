from scripts.plots.wandb_api import Api
from scripts.common import utils
import numpy as np
import os

class Metric:
    def __init__(self, label):
        self.label = label
        # runs x recorded points for metric
        self.data = []

    def append_run(self, run):
        self.data.append(run)

    def convert_data_to_np(self):
        # cut all lists to the same minimum length
        # but avoid runs that failed too quickly
        while True:
            lens = [len(values) for values in self.data]
            min_len = np.min(lens)
            max_len = np.max(lens)
            # when the minimum run length is too short, delete this run
            if (max_len - min_len) > 0.05 * max_len:
                # delete the run with the too short length
                index = lens.index(min_len)
                failed_run_data = self.data.pop(index)
                assert min_len == len(failed_run_data)
                print(f'Removed a run with min len of {min_len} where max is {max_len}')
            else: break

        data = [values[-min_len:] for values in self.data]
        self.data = np.array(data)

    def set_np_data(self, data):
        assert isinstance(data, np.ndarray)
        self.data = data

class Approach:
    def __init__(self, approach_name, project_name, run_name, metrics_names):
        self.name = approach_name
        self.project_name = project_name
        self.run_name = run_name
        self.path = utils.get_absolute_project_path() + f'graphs/{self.name}/'
        self.metrics_names = metrics_names
        self._api = Api(project_name)
        self._get_metrics_data()

    def _get_metrics_data(self):
        # first try to load from disc
        metrics_path = self.path + 'metrics.npz'
        if os.path.exists(metrics_path):
            self.metrics = []
            npz = np.load(metrics_path)
            for metric_name in self.metrics_names:
                metric = Metric(metric_name)
                metric.set_np_data(npz[metric_name])
                self.metrics.append(metric)
        # fetch from wandb if not on disc
        else:
            self.metrics = [Metric(name) for name in self.metrics_names]
            self._api.get_metrics(self)
            self._metrics_to_np()

    def _metrics_to_np(self):
        # convert each metric individually
        for metric in self.metrics:
            metric.convert_data_to_np()

    def save(self):
        # prepare and create path if necessary
        path = utils.get_absolute_project_path() + 'graphs/'
        path += self.name + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        assert isinstance(self.metrics[0].data, np.ndarray)
        metrics = [metric.data for metric in self.metrics]
        keys = [metric.label for metric in self.metrics]
        np.savez(path+'metrics', **{key:metric for key,metric in zip(keys, metrics)})

