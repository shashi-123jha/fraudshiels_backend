from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["fraudshield"]

users_collection = db["users"]
transactions_collection = db["transactions"]
