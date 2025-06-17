import os
import sys
import traceback
from .app import app

if __name__ == "__main__":
    # This entry point is used when running `python -m backend`
    app.logger.info('Starting Flask development server via `python -m backend`...')
    
    # Use environment variables for host and port, with defaults
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5002))
    # Use FLASK_DEBUG=1 environment variable to enable debug mode
    debug = True # Force debug mode for detailed error logging
    app.logger.info("Starting Flask development server with console logging. No file redirection.")
    # Ensure use_reloader is False to prevent issues with multiple log handlers
    # or unexpected restarts during debugging.
    try:
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    except SystemExit:
        print("DEBUG: Flask app run SystemExit caught in __main__.", file=sys.stderr)
        pass # Allow clean exit
    except Exception as e:
        print(f"DEBUG: EXCEPTION in __main__ during app.run: {type(e).__name__} - {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Optionally re-raise or exit with error code
        # raise
        sys.exit(1) # Indicate an error occurred
