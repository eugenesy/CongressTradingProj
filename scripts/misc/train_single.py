from src.train import train_window

if __name__ == "__main__":
    print("Starting Dry Run on Window 0...")
    try:
        results = train_window(window_id=0, epochs=100)
        print("\nDry Run Complete.")
        print(f"Final Metrics: {results}")
    except Exception as e:
        print(f"Training Failed: {e}")
        import traceback
        traceback.print_exc()
