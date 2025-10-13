#!/usr/bin/env python3
"""
Run script for the AdCampaign Agent.
Initializes Ray and starts the uvicorn server.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import dotenv
dotenv.load_dotenv()

import ray
import uvicorn


def main():
    """Initialize Ray and start the uvicorn server."""
    print("🚀 Starting AdCampaign Agent...")
    
    # Initialize Ray
    print("Initializing Ray...")
    try:
        ray.init()
        print("✅ Ray initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Ray: {e}")
        sys.exit(1)
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   Some features may not work properly.")
    
    # Start uvicorn server
    print("Starting uvicorn server...")
    try:
        uvicorn.run(
            "app.app:app",  # Import from app/app.py
            host="0.0.0.0",
            port=8014,
            reload=False,  # Disable reload to avoid Ray reinitialization issues
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
    finally:
        # Cleanup Ray
        try:
            ray.shutdown()
            print("✅ Ray shutdown complete")
        except Exception as e:
            print(f"⚠️  Ray shutdown error: {e}")


if __name__ == "__main__":
    main()
