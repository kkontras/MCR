# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ... import dice as D
from ... import material as M
from .base_weapon import Weapon


class BaseWhip(Weapon):
    pass


class Bullwhip(BaseWhip):

    def __init__(self):
        super().__init__(
            "bullwhip",
            weight=20,
            damage=D.Dice.from_str("1"),
            material=M.Leather,
            hit=0,
        )


class RubberHose(BaseWhip):

    def __init__(self):
        super().__init__(
            "rubber hose",
            weight=20,
            damage=D.Dice.from_str("d3"),
            material=M.Plastic,
            hit=0,
        )
