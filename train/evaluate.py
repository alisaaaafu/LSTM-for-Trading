"""
Model evaluation script
Author: Wanzhen Fu
"""
import lstm

if __name__ == "__main__":
    y, yhat = lstm.LSTMPredict()
    acc_total, acc_buy, acc_sell = lstm.PreAccuracy(y, yhat)
    print(f'Overall Accuracy: {acc_total:.2%}')
    print(f'Buy Signal (+1) Accuracy: {acc_buy:.2%}')
    print(f'Sell Signal (-1) Accuracy: {acc_sell:.2%}')
