from flask import Flask
import os # Moved import here

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'a_default_secret_key_change_me' # TODO: Change in production
    app.config['UPLOAD_FOLDER'] = 'uploads' # Directory to store uploaded files
    app.config['RESULTS_FOLDER'] = 'results' # Directory to store downloaded results
    app.config['DATABASE'] = 'experiments.db' # Path to the SQLite database

    # --- SSH Configuration (for connecting to existing instance) ---
    # These should ideally be loaded from environment variables or a config file
    app.config['SSH_HOSTNAME'] = os.environ.get('SSH_HOSTNAME', '') # IP Address or DNS
    app.config['SSH_USERNAME'] = os.environ.get('SSH_USERNAME', 'ubuntu')
    app.config['SSH_KEY_PATH'] = os.environ.get('SSH_KEY_PATH', os.path.expanduser('~/.ssh/id_rsa')) # Default to common key path

    # Ensure upload and results folders exist
    # import os # Removed from here
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULTS_FOLDER']):
        os.makedirs(app.config['RESULTS_FOLDER'])

    # Import and register blueprints/routes
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Initialize database
    from .core import db
    db.init_app(app)

    # The root route is now handled by the 'main' blueprint
    # @app.route('/')
    # def index():
    #     return "AI Training Tool - Coming Soon!"

    return app
