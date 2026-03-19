import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp import create_app
from webapp.extensions import db
from webapp.models import User


def main():
    parser = argparse.ArgumentParser(description="Initialize MySQL tables and optional admin user.")
    parser.add_argument("--admin", default="admin", help="Admin username")
    parser.add_argument("--password", default="admin123", help="Admin password")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        db.create_all()
        user = User.query.filter_by(username=args.admin).first()
        if user is None:
            user = User(username=args.admin)
            user.set_password(args.password)
            db.session.add(user)
            db.session.commit()
            print(f"Created admin user: {args.admin}")
        else:
            print(f"Admin user already exists: {args.admin}")


if __name__ == "__main__":
    main()
