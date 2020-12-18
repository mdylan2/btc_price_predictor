import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # USE THIS TO DISABLE ALL LOGGING INFORMATION FROM TENSORFLOW

from core.utils import *
from core.setup import set_up_app
from core.dataloader import DataLoader
from core.model import LSTMTimeSeriesModel
import pandas as pd

def main():
    '''
    Main app function
    '''
    # Trying to set up the app, catching any errors
    # Setting up logging, arg parsing, config file
    try:
        configs = set_up_app()
    except:
        logging.info("Something went wrong setting up the app. Exiting.")
        return
    
    if configs is None:
        logging.info("Something went wrong setting up the config file. Exiting.")
        return
    
    # Instantiating the dataloader in the core/dataloader module
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    
    # Train x and y
    x, y = data.get_train_data(
        lookback_window=configs['data']['sequence_length'],
        normalize=configs['data']['normalize']
    )
    
    # Instantiating the LSTMTimeSeriesModel model defined in the core/model module
    model = LSTMTimeSeriesModel()
    model.build_model(configs)

    # Training the model using the inbuilt method of the LSTMTimeSeriesModel class
    history = model.train(x,y,configs)
    
    # Plotting the train/val loss curves
    plot_training_curves(history)
    
    # Extracting the x and y test data
    x_test, y_test = data.get_test_data(
        lookback_window=configs['data']['sequence_length'],
        normalize=configs['data']['normalize']
    )
    
    # Predicting the results point by point
    predictions = model.predict_point_by_point(x_test)
    
    # Plotting the predictions compared to the actual values
    plot_results(predictions, y_test)
    
    # Using normalized predictions on test set, create a trading strategy
    signal = create_trading_strategy(predictions)
    
    # Combine this trading information with the test df. This is not the same as y_test
    # because y_test is smaller in size because of the lookback_window
    new_df = concatenate_strat_to_test(data.test_df, signal, configs["data"]["sequence_length"])
    
    # Compute the returns using this concatenated df
    new_df = compute_returns(new_df, configs["data"]["price_column"])
    
    # Compute Sharpe Ratio and CAGR for system and market, market being holding BTC
    system_metrics, market_metrics = compute_metrics(new_df)
    logging.info(f"Market CAGR: {system_metrics[0]*100:.1f}%")
    logging.info(f"Market Sharpe: {system_metrics[1]:.1f}")
    
    logging.info(f"System CAGR: {market_metrics[0]*100:.1f}%")
    logging.info(f"System Sharpe: {market_metrics[1]:.1f}")
    
    # Plot the results
    plot_returns(new_df)
    
# This script gets executed
if __name__ == '__main__':
    main()