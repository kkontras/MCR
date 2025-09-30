# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ... import dice as D
from ... import material as M
from .base_weapon import Weapon


class BaseBoomerang(Weapon):
    pass


class Boomerang(BaseBoomerang):

    def __init__(self):
        super().__init__("boomerang", weight=5, damage=D.Dice.from_str("d9"), material=M.Wood, hit=0)
