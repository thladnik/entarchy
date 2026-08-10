"""Turning a command line target into a SQLAlchemy URL.

Shared by the tools, all of which accept an entarchy directory, a SQLite file
or a URL in the same way.
"""
from __future__ import annotations

import os
import pathlib

import yaml


def _resolve_url(target: str) -> str:
    """Resolve a CLI target (directory, sqlite file or URL) into a SQLAlchemy URL."""

    path = pathlib.Path(target)

    # entarchy directory
    if path.is_dir():
        config_path = path / 'entarchy.yaml'
        if not config_path.exists():
            raise SystemExit(f'{target} is a directory but contains no entarchy.yaml')

        config = yaml.safe_load(open(config_path, 'r'))
        backend = config.get('backend', '')
        backend_config = config.get('backend_config', {})

        if 'sqlite' in backend.lower():
            db_path = (path / backend_config.get('dbname', 'entarchy.db')).as_posix()
            return f'sqlite:///{db_path}'

        if 'mysql' in backend.lower():
            from urllib.parse import quote_plus

            password = backend_config.get('dbpassword') or os.environ.get('ENTARCHY_DB_PASSWORD')
            if password is None:
                import getpass
                password = getpass.getpass(f'Password for {backend_config.get("dbuser")}: ')

            return (f'mysql+pymysql://{quote_plus(backend_config["dbuser"])}:{quote_plus(password)}'
                    f'@{backend_config["dbhost"]}/{backend_config["dbname"]}')

        raise SystemExit(f'Unknown backend {backend!r} in {config_path}')

    # sqlite database file
    if path.is_file():
        return f'sqlite:///{path.as_posix()}'

    # assume SQLAlchemy URL
    if '://' in target:
        return target

    raise SystemExit(f'Target {target!r} is neither an existing path nor a database URL')
