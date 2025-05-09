import sqlite3
import click
from flask import current_app, g
from flask.cli import with_appcontext

def get_db():
    """Connects to the application configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row # Return rows that behave like dicts

    return g.db

def close_db(e=None):
    """Closes the database connection."""
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    """Clear existing data and create new tables."""
    db = get_db()

    # Read schema from a file (or define inline)
    # Using a separate file is generally cleaner
    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    """Register database functions with the Flask app. This is called by
    the application factory.
    """
    app.teardown_appcontext(close_db) # Call close_db when cleaning up after returning the response
    app.cli.add_command(init_db_command) # Add the init-db command
