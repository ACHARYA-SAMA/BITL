"""Run the AR Hands desktop app with: python main.py."""
import sys

if __name__ == "__main__":
    try:
        from ar_hands.app import main
    except (ImportError, OSError) as exc:
        print("Dependencies could not load. Use Python 3.10-3.12 and run "
              f"python -m pip install -r requirements.txt\nDetails: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    raise SystemExit(main())
