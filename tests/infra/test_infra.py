import os
import time
import psycopg2
import redis
from minio import Minio
from minio.error import S3Error

def test_postgres():
    print("Testing Postgres connection...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database=os.getenv("POSTGRES_DB", "ai_server_db"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres")
        )
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result == (1,):
            print("[Postgres] Connection successful!")
        else:
            print(f"[Postgres] Unexpected result: {result}")
            return False
    except Exception as e:
        print(f"[Postgres] Connection failed: {e}")
        return False
    return True

def test_redis():
    print("Testing Redis connection...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        if r.ping():
            print("[Redis] Connection successful! - Ping: True")
            
            # Simple Write/Read test
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            if value == b'test_value':
                print("[Redis] Write/Read check passed.")
            else:
                print(f"[Redis] Write/Read check failed. Got {value}")
                return False
        else:
            print("[Redis] Ping failed.")
            return False
    except Exception as e:
        print(f"[Redis] Connection failed: {e}")
        return False
    return True

def test_minio():
    print("Testing MinIO connection...")
    try:
        client = Minio(
            "localhost:9000",
            access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            secure=False
        )
        # Check if we can list buckets (even if empty)
        buckets = client.list_buckets()
        print(f"[MinIO] Connection successful! - Found {len(buckets)} buckets.")
        
        # Try creating a test bucket
        bucket_name = "test-infra-bucket"
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"[MinIO] Created bucket '{bucket_name}'.")
        else:
            print(f"[MinIO] Bucket '{bucket_name}' already exists.")
            
        # Cleanup
        # client.remove_bucket(bucket_name) # Keep it for verification if needed
        
    except S3Error as e:
        print(f"[MinIO] S3 Error: {e}")
        return False
    except Exception as e:
        print(f"[MinIO] Connection failed: {e}")
        return False
    return True

if __name__ == "__main__":
    print("Starting Infrastructure Tests...")
    print("-" * 30)
    
    # Simple retry logic for initial startup
    max_retries = 5
    for i in range(max_retries):
        pg_ok = test_postgres()
        rd_ok = test_redis()
        mn_ok = test_minio()
        
        if pg_ok and rd_ok and mn_ok:
            print("-" * 30)
            print("All infrastructure tests passed!")
            exit(0)
        else:
            if i < max_retries - 1:
                print(f"Some tests failed. Retrying in 5 seconds... ({i+1}/{max_retries})")
                time.sleep(5)
            else:
                print("-" * 30)
                print("Infrastructure tests failed after retries.")
                exit(1)
