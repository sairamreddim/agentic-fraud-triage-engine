import random
from datetime import datetime, timedelta
import pandas as pd

def generate_large_dataset(num_rows=100000):
    print(f"[*] Generating {num_rows:,} mock transactions...")
    
    merchants = ["grocery", "electronics", "jewelry", "travel", "restaurant", "gas_station", "online_retail"]
    cards = [f"CARD-{str(i).zfill(3)}" for i in range(1, 501)] # 500 unique cards
    
    start_date = datetime.now() - timedelta(days=30)
    
    data = []
    for i in range(1, num_rows + 1):
        txn_id = f"TXN-{100000 + i}"
        card_id = random.choice(cards)
        # Skew amounts: mostly small, occasional large amounts
        amount = round(random.expovariate(1/50) + random.choice([5.0, 15.0, 120.0]), 2)
        if amount > 5000: 
            amount = 499.99 # cap extreme outliers unless intentional
            
        merchant = random.choice(merchants)
        # Random timestamp within the last 30 days
        random_seconds = random.randint(0, 30 * 86400)
        timestamp = (start_date + timedelta(seconds=random_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        
        data.append({
            "transaction_id": txn_id,
            "card_id": card_id,
            "timestamp": timestamp,
            "amount": amount,
            "merchant_category": merchant
        }
    )
    
    df = pd.DataFrame(data)
    df.to_csv("large_transactions.csv", index=False)
    print(f"[+] Successfully generated 'large_transactions.csv' with {num_rows:,} rows!")

if __name__ == "__main__":
    generate_large_dataset(100000)