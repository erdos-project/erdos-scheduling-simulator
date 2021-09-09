from absl import app, flags

from data_loader import TaskDataLoader
from simulator import FifoSimulator

FLAGS = flags.FLAGS

flags.DEFINE_string('json_profile_file', 'data/pylot_profile.json',
                    'Path to the Pylot JSON log file')


def main(args):
    tasks = TaskDataLoader(FLAGS.json_profile_file, verbose=1)

    ops = tasks.get_ops_converter()

    print("=" * 20)
    print(f"Operators: {ops}")

    print("=" * 20)
    b = [(entry['name'], entry['args'], entry['ts'])
         for entry in tasks.get_data_as_binned()[-2]]
    print(f"Second to last timestamp: {b}")

    print("=" * 20)
    print("First data entry: {tasks.get_data()[0]}")

    tasks.get_task_list()[0:5]

    print("=" * 20)
    print(f"Lattice dependencies: {tasks.get_lattice()}")

    print("=" * 20)
    s = FifoSimulator(10,
                      10,
                      tasks.get_task_list()[0:100],
                      tasks.get_lattice(),
                      gpu_exact_match=True)
    print(
        f"Task GPU requirement: {[e.needs_gpu for e in tasks.get_task_list()]}"
    )

    print("=" * 20)
    print(s.worker_pool)

    steps = 8537824 + 100
    print("=" * 20)
    print(f"Simulate for {steps} steps")
    print("=" * 20)
    s.simulate(steps)

    print("=" * 20)
    print(f"Tasks pending: {s.tasks_list}")
    print(f"Tasks running: {s.show_running_tasks()}")


if __name__ == '__main__':
    app.run(main)
