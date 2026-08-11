"""Demo user store (replace with a real database in production)."""

from . import auth

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": auth.get_password_hash("secret"),
        "disabled": False,
    }
}
