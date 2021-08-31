import json
from functools import reduce

from workload.lattice import Lattice
from workload.operator import Operator
from workload.task import Task


class OpsConverter:
    """Extracts the operator name,id map and dependency graph."""
    def __init__(self, data, needs_gpu):
        unique_ops = list({entry['name'] for entry in data})
        unique_ops.sort()
        self.ops = [(i, op, needs_gpu(op)) for i, op in enumerate(unique_ops)]
        self.count = len(self.ops)
        self.id_to_ops = {i: op for i, op, _ in self.ops}
        self.ops_to_id = {op: i for i, op, _ in self.ops}
        self.ids_to_needs_gpu = {i: needs_gpu for i, _, needs_gpu in self.ops}
        self.dep_matrix = [[False] * self.count] * self.count

    def __repr__(self):
        return str(self.ops)

    def get_all_ops(self):
        return self.ops

    def get_id(self, name: str):
        return self.ops_to_id[name]

    def get_op(self, op_id: int):
        return self.id_to_ops[op_id]

    def get_gpu(self, name: str):
        return self.ids_to_needs_gpu[self.get_id(name)]

    def get_deps(self):
        """Returns the dependency matrix."""
        return self.dep_matrix


class TaskDataLoader:
    """
    Args:
        path: Path to the Pylot profile json file.
        verbose: Greater than 0 if verbose logging should be enabled.
    """
    def __init__(self, path: str, verbose: int = 0):
        with open(path, ) as f:
            data = json.load(f)
        start_time = reduce(lambda x, y: x
                            if x['ts'] < y['ts'] else y, data)['ts']
        for entry in data:
            ts = 0
            if entry['args']['timestamp'] != '[]':
                ts = int(entry['args']['timestamp'][1:-1])
            entry['args']['timestamp'] = ts
            entry['ts'] -= start_time
        self.data = data
        if verbose > 0:
            print(f"Times are relative to UNIX time: {start_time}; \n first "
                  "task in replay starts at {data[0]['ts']} and end at "
                  "{data[-1]['ts']}")
            print(f"There are {len(data)} tasks.")
        ids = list({entry['args']['timestamp'] for entry in data})
        ids.sort()
        self.ids = ids
        if verbose > 0:
            print(f" There are {len(ids)} logical time labels ending "
                  "with {ids[-1]}")
        # Object detection and lane detection require a GPU.
        self.ops_converter = OpsConverter(
            data, lambda name: True
            if ('rcnn' in name or 'lanenet' in name) else False)
        # Create tasks from the json entries.
        self.task_from_log = [
            Task(idx,
                 self.ops_converter.get_id(entry['name']),
                 entry['ts'],
                 runtime=entry['dur'],
                 deadline=None,
                 needs_gpu=self.ops_converter.get_gpu(entry['name']))
            for idx, entry in enumerate(data)
        ]
        self.lattice = Lattice([
            Operator(idx, [], 10, needs_gpu=needs_gpu, name=name)
            for idx, name, needs_gpu in self.ops_converter.get_all_ops()
        ])

    def get_data(self):
        return self.data

    def get_data_as_binned(self):
        # prolly use this to build dependency matrix; for now online computed cause it's expensive
        return [[
            entry for entry in self.data if entry['args']['timestamp'] == ts
        ] for ts in self.ids]

    def get_ops_converter(self):
        return self.ops_converter

    def get_task_list(self):
        return self.task_from_log

    def get_lattice(self):
        return self.lattice
