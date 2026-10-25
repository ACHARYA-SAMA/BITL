"""Gesture engine: geometry features, recognition, stabilization, motion, events.

Import from the submodules directly, for example
``from ar_hands.gestures.engine import GestureEngine``. This file deliberately
imports nothing: ``ar_hands.config`` needs ``gestures.types`` for its defaults,
while the engine needs ``ar_hands.config``, so eager imports here would create
an import cycle.
"""
