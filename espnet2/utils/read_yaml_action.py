import argparse
import copy

import yaml


class ReadYAMLAction(argparse.Action):
    """Action class to store everything in the given yaml file.

    Examples:
        >>> parser = argparse.ArgumentParser()
        # TODO
    """

    _syntax = """Syntax:
  {op} <key>=<yaml-string>
  {op} <key>.<key2>=<yaml-string>
  {op} <python-dict>
  {op} <yaml-string>
e.g.
  {op} a=4
  {op} a.b={{c: true}}
  {op} {{"c": True}}
  {op} {{a: 34.5}}
"""

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            type=str,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_strings=None):
        # --{option} file.yaml -> { ... contents of yaml ... }

        # TODO: Handle errors

        config_file = values
        with open(config_file, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        new_items = { k : args[k] for k in set(args) - set(vars(namespace)) }

        for k, v in new_items.items():
            setattr(namespace, k, v)
