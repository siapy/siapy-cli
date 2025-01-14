from collections import defaultdict
from typing import Optional

from rich import print as RichPrint
from rich.text import Text as RichText
from source.analysis.metrics import METRIC_FUNC, Metrics
from source.analysis.params import DirParams, params_from_path
from tabulate import tabulate


def generate_metrics_table(metrics: list[Metrics]):
    headers = ["ID"] + list(METRIC_FUNC.keys())
    grouped_metrics = defaultdict(list)
    for metric in metrics:
        grouped_metrics[metric.meta_id].append(
            f"{metric.mean:.2f} (+- {metric.std:.2f})"
        )
    rows = [[meta_id] + values for meta_id, values in grouped_metrics.items()]
    table = tabulate(rows, headers, tablefmt="grid")
    metrics_all = metrics[-len(METRIC_FUNC.keys()) :]
    return table, metrics_all


def display_metrics(
    metrics: dict[str, list[Metrics]],
    model: Optional[str] = None,
    do_optimize: Optional[bool] = None,
    data_loader: Optional[str] = None,
):
    def check_filter(params: DirParams) -> bool:
        if model is not None and params.estimator_name != model:
            return False
        if do_optimize is not None and params.estimator_is_optimized != do_optimize:
            return False
        if data_loader is not None and params.data_loader_name != data_loader:
            return False
        return True

    headers = ["ID"] + list(METRIC_FUNC.keys())
    rows = []
    for key, metrics_ in metrics.items():
        params = params_from_path(key)
        if not check_filter(params):
            continue
        row = [key]
        for metric in metrics_:
            row.append(str(metric.mean))
        rows.append(row)

    table = tabulate(rows, headers, tablefmt="grid")
    RichPrint(RichText(table, style="white"))
