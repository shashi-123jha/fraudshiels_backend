from db import users_collection

def create_user(user):
    return users_collection.insert_one(user)

def find_user_by_email(email):
    return users_collection.find_one({"email": email})
