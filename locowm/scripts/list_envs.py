"""
Script to print all the available environments in the extension.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.
"""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:  # Support the legacy ``python locowm/scripts/list_envs.py`` form.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

parser = argparse.ArgumentParser(description="List the registered LocoWM environments.")
parser.parse_args()

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import locowm

locowm.ensure_runtime()


import gymnasium as gym
from prettytable import PrettyTable


def main():
    """Print the registered Isaac environments."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available LocoWM Environments"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if task_spec.id.startswith("Isaac-"):
            # add details to table
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close the app
        simulation_app.close()
