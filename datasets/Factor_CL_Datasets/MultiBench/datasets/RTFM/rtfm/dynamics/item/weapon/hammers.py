# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ... import dice as D
from ... import material as M
from .base_weapon import Weapon


class BaseHammer(Weapon):
    pass


class WarHammer(BaseHammer):

    def __init__(self):
        super().__init__(
            "war hammer",
            weight=50,
            damage=D.Dice.from_str("d4"),
            material=M.Iron,
            hit=0,
        )
