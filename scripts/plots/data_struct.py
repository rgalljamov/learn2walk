from scripts.plots.wandb_api import Api
from scripts.common import utils
import numpy as np
import os

class Metric:
    def __init__(self, label, approach_name='approach', train_duration_mio=None):
        self.label = label
        self.approach_name = approach_name
        self.train_duration_mio = train_duration_mio
        # runs x recorded points for metric
        self.data = []

    def set_approach_name(self, approach_name):
        self.approach_name = approach_name

    def append_run(self, run):
        self.data.append(run)

    def convert_data_to_np(self):
        # to account for scalar values like steps_to_conv
        if not (isinstance(self.data[0], list) or isinstance(self.data[0], np.ndarray)):
            self.data = np.array(self.data)
            return
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

    def calculate_statistics(self):
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        if isinstance(self.mean, np.ndarray) and len(self.mean) > 10:
            self.mean_fltrd = utils.smooth_exponential(self.mean, 0.005)
            self.std_fltrd = utils.smooth_exponential(self.std, 0.005)
        else: self.mean_fltrd = self.mean


class Approach:
    def __init__(self, approach_name, project_name=None, run_name=None, metrics_names=[]):
        self.name = approach_name
        self.project_name = project_name
        self.train_duration_mio = 16 if 'pd' in approach_name else 8
        self.run_name = run_name
        self.path = utils.get_absolute_project_path() + f'graphs/{self.name}/'
        self.metrics_names = metrics_names
        self._get_metrics_data()
        self._calculate_statistics()

    def _get_metrics_data(self):
        # first try to load from disc
        metrics_path = self.path + 'metrics.npz'
        if os.path.exists(metrics_path):
            from scripts.plots.compare import MET_SUM_SCORE
            from scripts.common.callback import EVAL_INTERVAL_RARE
            self.metrics = []
            npz = np.load(metrics_path)
            for metric_label in npz.keys():
                self.metrics_names.append(metric_label)
                metric = Metric(metric_label, self.name, self.train_duration_mio)
                metric_data = npz[metric_label]
                # normalize summary score
                if metric_label == MET_SUM_SCORE:
                    max_score = self.train_duration_mio*1e6/EVAL_INTERVAL_RARE
                    metric_data /= 0.5*max_score
                    # normalize training duration to range [0,1]
                    metric_data *= 16/self.train_duration_mio
                    metric_data *= 100 # show in percent
                metric.set_np_data(metric_data)
                self.metrics.append(metric)
        # fetch from wandb if not on disc
        else:
            self._api = Api(self.project_name)
            self.metrics = [Metric(name, self.name, self.train_duration_mio) for name in self.metrics_names]
            self._api.get_metrics(self)
            self._metrics_to_np()

    def _calculate_statistics(self):
        for metric in self.metrics:
            metric.calculate_statistics()

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
        print('Successfully saved approach:', self.name)

