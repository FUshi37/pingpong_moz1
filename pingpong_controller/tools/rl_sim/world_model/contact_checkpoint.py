"""Narrow source compatibility for the unchanged legacy contact calculation."""

from __future__ import annotations


def legacy_geometry_dispatch_compatible(old_sources: dict, new_sources: dict) -> bool:
    """Allow only the audited import/docstring change with exact old dispatch.

    This is not a general permission to resume after a contact-model edit.
    New source edits or native-geometry selection need their own checkpoint.
    """
    return (
        old_sources.get('contact_model.py') ==
        '32e3cbc9a9cebd106fd9dd55bd494808053d9fab69e4cc76670138e0f6a0e8a0'
        and 'contact_geometry.py' not in old_sources
        and new_sources.get('contact_model.py') ==
        'ccd94659584f832c6ac170bd6a576a37f07439e1b4a985ff61141ac4107e4247'
        and new_sources.get('contact_geometry.py') ==
        'd7ddfd72d0e6fecb9bbbca5b4dc1653a0b47d81dc1e097fe20f39e24cb4bc886'
    )
