"""Read-only integrity checks for preserved, opt-in training execution versions."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re


def _sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _file_below(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or Path(relative).is_absolute():
        raise ValueError('Version file must stay within its declared directory')
    return path


def verify_version(
    registry_dir: Path, version: str, repo_root: Path, *,
    check_current: bool = False, check_installed: bool = False,
) -> dict:
    """Verify saved sources and optionally the active source/environment."""
    if not re.fullmatch(r'[a-z][a-z0-9_]+', version):
        raise ValueError('Invalid execution version name')
    catalog = json.loads((registry_dir / 'versions.json').read_text())
    if catalog['schema_version'] != 1 or catalog['default'] != 'legacy':
        raise ValueError('Unsupported catalog or changed opt-in default')
    entry = catalog['versions'][version]
    manifest_path = _file_below(registry_dir, entry['manifest'])
    if _sha256(manifest_path) != entry['manifest_sha256']:
        raise ValueError('Preserved version manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['schema_version'] != 1 or manifest['version'] != version
            or manifest['selector'] != entry['selector']
            or manifest['default_enabled'] is not False
            or manifest['auto_enable_on_resume'] is not False):
        raise ValueError('Version identity or opt-in contract changed')
    errors = []
    checked = []
    for record in manifest['files']:
        expected = record['sha256']
        if not re.fullmatch(r'[a-f0-9]{64}', expected):
            raise ValueError('Malformed preserved source hash')
        snapshot = _file_below(manifest_path.parent, record['snapshot'])
        if _sha256(snapshot) != expected:
            errors.append('Saved source hash mismatch: ' + record['snapshot'])
        if check_current:
            current = _file_below(repo_root, record['repository'])
            if _sha256(current) != expected:
                errors.append('Current source differs: ' + record['repository'])
        checked.append(record['repository'])
    if check_installed:
        for package, expected in manifest['dependencies']['packages'].items():
            if metadata.version(package) != expected:
                errors.append('Installed package version differs: ' + package)
        for record in manifest['dependencies']['native_sources']:
            distribution = metadata.distribution(record['package'])
            path = Path(distribution.locate_file(record['path']))
            if _sha256(path) != record['sha256']:
                errors.append('Installed native source differs: ' + record['path'])
    return dict(ok=not errors, version=version, selector=manifest['selector'],
                default_enabled=False, checked_files=checked,
                current_source_checked=check_current,
                installed_dependencies_checked=check_installed, errors=errors)


def main() -> None:
    """Check a saved version without importing JAX or starting training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='mjx_topology_execution_v2')
    parser.add_argument('--check-current', action='store_true')
    parser.add_argument('--check-installed', action='store_true')
    opts = parser.parse_args()
    sim_dir = Path(__file__).resolve().parent
    try:
        result = verify_version(
            sim_dir / 'execution_versions', opts.version, sim_dir.parents[2],
            check_current=opts.check_current,
            check_installed=opts.check_installed)
    except (OSError, ValueError, KeyError, metadata.PackageNotFoundError) as error:
        result = dict(ok=False, version=opts.version, errors=[str(error)])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
