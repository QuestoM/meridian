"""Keep a re-exported name substitutable, not merely resolvable, after the split.

Frozen. Owned by W0-1, declared in ``docs/ux-gauntlet/contracts/W0-1.md``.

Before the wave-zero router split, dashboard_api, insights_api and catalog_api
each held their routes, their builders, their loaders and their path constants in
one namespace, so substituting a name on the module reached every reader of it.
That is how several frozen tests measure a cache key: they substitute
``SETTINGS_PATH`` or ``DATA_DIR`` on the module and then assert that touching the
file invalidates the cache. Another substitutes a builder for one that raises and
asserts the cold overview never calls it.

After the split those readers live in the per-owner modules. A plain re-export
still resolves the name, but a substitution on the old module would no longer
reach them, which turns a real measurement into a silent no-op. This module
mirrors a write to a re-exported name into every module that inherited it, so a
substitution behaves as it did before the split. Reads are untouched: only writes
are mirrored, and only into the modules the layer declares.

A mirror target is found two ways, so a name that lost its leading underscore on
the move is still reached: by the same name, and by the same object. Object
matching is skipped for atomic immutables, where identity would be an accident of
interning rather than the same thing.

One consequence is stated rather than hidden: where two pre-split modules
contributed routes to one new module, a substitution now reaches both of those
routes. `/api/schedule` and `/api/inventory` both live in ``week_api``, so
substituting ``DATA_DIR`` on either compatibility layer reaches both. No probe in
the suite depends on the narrower behaviour, and the alternative, splitting the
new modules by their pre-split origin rather than by their owner, is exactly what
this wave exists to stop doing.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Iterable

# Never mirrored: a compatibility layer's own identity and its own router.
_NEVER_MIRRORED = frozenset({"logger", "router", "_mirror_names", "_mirror_map"})

# Identity says nothing about these: two modules can hold the same short string
# or small integer without sharing anything. They are matched by name only.
_ATOMIC = (str, bytes, int, float, bool, type(None), tuple, frozenset)


class _MirroringModule(ModuleType):
    """A module whose writes to declared names are mirrored into other modules."""

    _mirror_names: frozenset[str] = frozenset()
    _mirror_map: dict[str, tuple[tuple[ModuleType, str], ...]] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in self._mirror_names:
            for target, target_name in self._mirror_map.get(name, ()):
                setattr(target, target_name, value)


def _targets_for(name: str, value: Any, targets: tuple[ModuleType, ...]) -> tuple:
    found: list[tuple[ModuleType, str]] = []
    for target in targets:
        for target_name, target_value in vars(target).items():
            if target_name.startswith("__"):
                continue
            same_name = target_name == name
            same_object = not isinstance(value, _ATOMIC) and target_value is value
            if same_name or same_object:
                found.append((target, target_name))
    return tuple(dict.fromkeys(found))


def mirror_writes(module_name: str, targets: Iterable[ModuleType]) -> None:
    """Mirror writes to every re-exported name of ``module_name`` into ``targets``.

    Call it as the last statement of a compatibility layer, once every name it
    re-exports is bound. The mirrored set is that namespace minus the layer's own
    identity and minus the modules it imported, so a name added to the layer later
    is covered without a second list to keep in step.
    """
    module = sys.modules[module_name]
    targets = tuple(targets)
    mirror_map: dict[str, tuple[tuple[ModuleType, str], ...]] = {}
    for name, value in vars(module).items():
        if name.startswith("__") or name in _NEVER_MIRRORED or isinstance(value, ModuleType):
            continue
        found = _targets_for(name, value, targets)
        if found:
            mirror_map[name] = found
    module.__class__ = _MirroringModule
    module._mirror_names = frozenset(mirror_map)
    module._mirror_map = mirror_map
