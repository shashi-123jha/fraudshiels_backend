from db import transactions_collection

def create_transaction(txn):
    return transactions_collection.insert_one(txn)

def get_transactions_by_user(user_id):
    return list(transactions_collection.find({"user_id": user_id}, {"_id": 0}))
