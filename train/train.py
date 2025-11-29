"""
Training script for LSTM trading model
Author: Wanzhen Fu
"""
import lstm
import time

if __name__ == "__main__":
    begin_time = time.time()
    print("Training started...")
    lstm.TrainLSTM()
    end_time = time.time()
    print(f"Training completed in {end_time-begin_time:.2f}s")
