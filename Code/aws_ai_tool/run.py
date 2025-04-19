from app import create_app

app = create_app()

if __name__ == '__main__':
    # Note: debug=True is convenient for development but should be False in production.
    # host='0.0.0.0' makes the server accessible on the local network.
    app.run(debug=True, host='0.0.0.0', port=5000)
