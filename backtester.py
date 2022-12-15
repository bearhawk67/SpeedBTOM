from ctypes import *
from models import *
import typing
from database import Hdf5Client
from utils import *
from os.path import exists as file_exists
import pandas as pd
import numpy as np
import strategies.obv
import strategies.ichimoku
import strategies.support_resistance
import strategies.mfi
import strategies.guppy
import random
import os.path
import time


def run(contract: Contract, strategy: str, tf: str, from_time: int, to_time: int, initial_capital: int, file_name: str):
    exchange = "bybit"
    param_descriptions = STRAT_PARAMS[strategy]
    params = dict()
    csv_data = pd.read_csv(file_name, header=0)
    csv_data = csv_data.iloc[:, 1:]
    rows = csv_data.index[-1] + 1
    multitest_results = pd.DataFrame()
    file_name = file_name.lstrip("Results/Optimizations/OptimizerResults_")
    file_name = file_name.rstrip(".csv")

    iterations = 0
    while iterations < rows:
        for p_code, p in param_descriptions.items():
            if p["type"] == Y_N:
                params[p_code] = str(csv_data.at[iterations, p_code])
            else:
                params[p_code] = p["type"](csv_data.at[iterations, p_code])

    # if strategy == "obv":
    #     h5_db = Hdf5Client()
    #     data = h5_db.get_data(contract, from_time, to_time)
    #     data = resample_timeframe(data, tf)
    #
    #     pnl, max_drawdown = strategies.obv.backtest(data, ma_period=params["ma_period"])
    #     return pnl, max_drawdown
    #
    # elif strategy == "ichimoku":
    #     h5_db = Hdf5Client()
    #     data = h5_db.get_data(contract, from_time, to_time)
    #     data = resample_timeframe(data, tf)
    #
    #     pnl, max_drawdown = strategies.ichimoku.backtest(data, tenkan_period=params["tenkan"],
    #                                                      kijun_period=params["kijun"])
    #     return pnl, max_drawdown
    #
    # elif strategy == "sup_res":
    #     h5_db = Hdf5Client()
    #     data = h5_db.get_data(contract, from_time, to_time)
    #     data = resample_timeframe(data, tf)
    #
    #     pnl, max_drawdown = strategies.support_resistance.backtest(data, min_points=params["min_points"],
    #                                                                min_diff_points=params["min_diff_points"],
    #                                                                rounding_nb=params["rounding_nb"],
    #                                                                take_profit=params["take_profit"],
    #                                                                stop_loss=params["stop_loss"])
    #     return pnl, max_drawdown
    #
    # elif strategy == "mfi":
    #     h5_db = Hdf5Client()
    #     data = h5_db.get_data(contract, from_time, to_time)
    #     data = resample_timeframe(data, tf)
    #     pnl, max_drawdown = strategies.mfi.backtest(data, period=params['period'], multiplier=params['multiplier'],
    #                                                 ypos=params['ypos'])
    #     return pnl, max_drawdown
    #
    # elif strategy == "sma":
    #     # import C++ library
    #     lib = get_library()
    #
    #     obj = lib.Sma_new(exchange.encode(), contract.symbol.encode(), tf.encode(), from_time, to_time)
    #     lib.Sma_execute_backtest(obj, params["slow_ma"], params["fast_ma"])
    #     pnl = lib.Sma_get_pnl(obj)
    #     max_drawdown = lib.Sma_get_max_dd(obj)
    #
    #     return pnl, max_drawdown

        h5_db = Hdf5Client()
        data = h5_db.get_data(contract, from_time, to_time)
        data = resample_timeframe(data, tf)
        pnl, max_drawdown, win_rate, rr_long, rr_short, num_trades, mod_win_rate, max_losses, max_wins, num_longs, \
            num_shorts \
            = strategies.guppy.backtest(df=data, initial_capital=initial_capital,
                                        trade_longs=params['trade_longs'],
                                        trade_shorts=params['trade_shorts'], sl_long=params['sl_long'],
                                        sl_short=params['sl_short'], mfi_long=params['mfi_long'],
                                        mfi_short=params['mfi_short'], mfi_period=params['mfi_period'],
                                        mfi_mult=params['mfi_mult'], mfi_ypos=params['mfi_ypos'],
                                        mfi_long_threshold=params['mfi_long_threshold'],
                                        mfi_short_threshold=params['mfi_short_threshold'],
                                        adx_long=params['adx_long'], adx_short=params['adx_short'],
                                        macd_short=params['macd_short'], macd_fast=params['macd_fast'],
                                        macd_slow=params['macd_slow'], macd_signal=params['macd_signal'],
                                        macd_long=params['macd_long'], rsi_long=params['rsi_long'],
                                        rsi_short=params['rsi_short'], rsi_length=params['rsi_length'],
                                        rsi_overbought=params['rsi_overbought'], rsi_oversold=params['rsi_oversold'],
                                        ema200_long=params['ema200_long'],
                                        ema200_short=params['ema200_short'],
                                        guppy_fast_long=params['guppy_fast_long'],
                                        guppy_fast_short=params['guppy_fast_short'],
                                        ribbon_check_long=params['ribbon_check_long'],
                                        ribbon_check_short=params['ribbon_check_short'],
                                        move_sl_type_long=params['move_sl_type_long'],
                                        move_sl_type_short=params['move_sl_type_short'],
                                        move_sl_long=params['move_sl_long'],
                                        move_sl_short=params['move_sl_short'], risk=params['risk'],
                                        leverage=params['leverage'], tp_long=params['tp_long'],
                                        tp_short=params['tp_short'], ltp1=params['ltp1'],
                                        ltp1_qty=params['ltp1_qty'], ltp2=params['ltp2'],
                                        ltp2_qty=params['ltp2_qty'], ltp3=params['ltp3'],
                                        stp1=params['stp1'], stp1_qty=params['stp1_qty'],
                                        stp2=params['stp2'], stp2_qty=params['stp2_qty'],
                                        stp3=params['stp3'], mode="b", contract=contract, tf=tf,
                                        from_time=from_time, to_time=to_time,
                                        bb_long=params['bb_long'], bb_short=params['bb_short'],
                                        bb_length=params['bb_length'],
                                        bb_mult=params['bb_mult'],
                                        wae_long=params['wae_long'], wae_short=params['wae_short'],
                                        wae_sensitivity=params['wae_sensitivity'],
                                        wae_fast_length=params['wae_fast_length'],
                                        wae_slow_length=params['wae_slow_length'],
                                        wae_bb_length=params['wae_bb_length'],
                                        wae_bb_mult=params['wae_bb_mult'],
                                        wae_rma_length=params['wae_rma_length'],
                                        wae_dz_mult=params['wae_dz_mult'],
                                        wae_expl_check=params['wae_expl_check'],
                                        adx_smoothing=params['adx_smoothing'],
                                        adx_di_length=params['adx_di_length'],
                                        adx_length_long=params['adx_length_long'],
                                        adx_length_short=params['adx_length_short'], iteration=iterations)
        iterations += 1
        # return pnl, max_drawdown, win_rate, rr_long, rr_short, num_trades, max_losses, max_wins


def random_start_end(contract: Contract, tf: str, time_delta: float, type:str) -> typing.Tuple[int, int]:
    h5_db = Hdf5Client()
    data_start, data_end = h5_db.get_first_last_timestamp(contract)
    if type == "f":
        period_start = random.randint((data_start + (60 * TF_SECONDS[tf])), (data_end - time_delta))
        period_start -= (60 * TF_SECONDS[tf])
        period_end = int(period_start + time_delta)
    elif type == "l":
        period_start = random.randint((data_end - 31536000), (data_end - time_delta))
        period_start -= (60 * TF_SECONDS[tf])
        period_end = int(period_start + time_delta)
    return period_start, period_end


def multitest(contract: Contract, strategy: str, tf: str, test_type: str, time_delta: int, initial_capital: int,
              tests: int, pool: str, file_name: str):
    exchange = "bybit"
    param_descriptions = STRAT_PARAMS[strategy]
    params = dict()
    # if test_type == "s":
    #     input_mode = ["manual", "from csv"]
    #     while True:
    #         mode = input(f"Parameter input mode ({', '.join(input_mode)}): ").lower()
    #         if mode in input_mode:
    #             break
    #
    #     if mode == "manual":
    #         for p_code, p in param_descriptions.items():
    #             while True:
    #                 if p["type"] == Y_N:
    #                     params[p_code] = str(input(p["name"] + ": "))
    #                     if params[p_code] in Y_N:
    #                         break
    #                 else:
    #                     try:
    #                         params[p_code] = p["type"](input(p["name"] + ": "))
    #                         break
    #                     except ValueError:
    #                         continue
    #     else:
    #         while True:
    #             file_name = str(input("Input CSV file name to read parameters from (including .csv): "))
    #
    #             if file_exists(file_name):
    #                 break
    #             else:
    #                 print(f"ERROR: {file_name} does not exist")
    #                 continue
    #
    #     multitest_start_time = time.time()
    #     csv_data = pd.read_csv(file_name, header=None, names=["parameter", "value"], index_col="parameter")
    #     for p_code, p in param_descriptions.items():
    #         if p["type"] == Y_N:
    #             params[p_code] = str(csv_data.at[str(p_code), "value"])
    #         else:
    #             params[p_code] = p["type"](csv_data.at[str(p_code), "value"])
    #     tester(contract, strategy, tf, days, hours, tests, initial_capital, params, csv_data, test_type, 0, pool)

    # elif test_type == "mp":
    #     while True:
    #         file_name = str(input("Input CSV file name to read parameters from (including .csv): "))
    #
    #         if file_exists(file_name):
    #             break
    #         else:
    #             print(f"ERROR: {file_name} does not exist")
    #             continue
    #
    #     # Minimum Average PNL to keep results
    #     while True:
    #         min_avg_pnl = input("Minimum acceptable Average PNL or Press Enter to not set a minimum: ")
    #         try:
    #             if min_avg_pnl == "":
    #                 min_avg_pnl = -100
    #                 break
    #             else:
    #                 min_avg_pnl = int(min_avg_pnl)
    #                 break
    #         except ValueError:
    #             continue
    #
    #     # Minimum % Positive to keep results
    #     while True:
    #         min_percent_positive = input("Minimum acceptable % Positive or Press Enter to not set a minimum: ")
    #         try:
    #             if min_percent_positive == "":
    #                 min_percent_positive = 0
    #                 break
    #             else:
    #                 min_percent_positive = int(min_percent_positive)
    #                 break
    #         except ValueError:
    #             continue
    min_percent_positive = 0
    min_avg_pnl = -100
    multitest_start_time = time.time()

    csv_data = pd.read_csv(file_name, header=0)
    csv_data = csv_data.iloc[:, 1:]
    rows = csv_data.index[-1] + 1
    print(f"{rows} Results")
    multitest_results = pd.DataFrame()
    file_name = file_name.lstrip("Results/Optimizations/OptimizerResults_")
    file_name = file_name.rstrip(".csv")

    iterations = 0
    while iterations < rows:
        for p_code, p in param_descriptions.items():
            if p["type"] == Y_N:
                params[p_code] = str(csv_data.at[iterations, p_code])
            else:
                params[p_code] = p["type"](csv_data.at[iterations, p_code])
                # print(f"{iterations} {p_code}: {params[p_code]}")
        results = tester(contract, strategy, tf, time_delta, tests, initial_capital, params, csv_data,
                         test_type, iterations, pool, file_name)
        results = results.drop(results.index[:-1])
        results = results.reset_index()
        results = results.iloc[:, 1:]
        if (results['pnl_avg'][0] >= min_avg_pnl) and (results['%_positive'][0] >= min_percent_positive):
            results['optimizer_file_row'] = iterations + 2
            multitest_results = pd.concat([multitest_results, results], axis=0)
        iterations += 1
    print("\n")

    # print(f"Time to complete: {int(time.time() - multitest_start_time)} seconds")
    if os.path.exists(f"Results/UBERs/UBERMultitestResults_{file_name}_{tests}_tests.csv"):
        while True:
            try:
                myfile = open(f"Results/UBERs/UBERMultitestResults_{file_name}_{tests}_tests.csv",
                              "w+")
                break
            except IOError:
                input(f"Cannot write results to csv file. Please close \n"
                      f"Results/UBERs/UBERMultitestResults_{file_name}_{tests}_tests.csv"
                      f"\nThen press Enter to retry.")
    multitest_results.to_csv(f"Results/UBERs/UBERMultitestResults_{file_name}_{tests}_tests.csv")


def tester(contract: Contract, strategy: str, tf: str, time_delta, tests: int, initial_capital: int,
           params: typing.Dict, csv_data: pd.DataFrame, mode: str, iteration: int, pool: str, file_name: str) -> \
        pd.DataFrame:

    df = pd.DataFrame()
    pnl_list = []
    pnl_avg = []
    pnl_std = []
    pnl_cv = []
    max_dd_list = []
    max_dd_avg = []
    max_dd_std = []
    max_dd_cv = []
    win_rate_list = []
    win_rate_avg = []
    win_rate_std = []
    win_rate_cv = []
    mod_win_rate_list = []
    mod_win_rate_avg = []
    mod_win_rate_std = []
    mod_win_rate_cv = []
    num_trades_list = []
    num_trades_avg = []
    num_trades_std = []
    num_trades_cv = []
    max_losses_list = []
    max_losses_avg = []
    max_losses_std = []
    max_losses_cv = []
    max_wins_list = []
    max_wins_avg = []
    max_wins_std = []
    max_wins_cv = []
    trades_won_list = []
    trades_won_avg = []
    trades_won_std = []
    trades_won_cv = []
    trades_lost_list = []
    trades_lost_avg = []
    trades_lost_std = []
    trades_lost_cv = []
    breakeven_trades_list = []
    breakeven_trades_avg = []
    breakeven_trades_std = []
    breakeven_trades_cv = []
    num_longs_list = []
    num_longs_avg = []
    num_longs_std = []
    num_longs_cv = []
    num_shorts_list = []
    num_shorts_avg = []
    num_shorts_std = []
    num_shorts_cv = []
    profit_factor_list = []
    profit_factor_avg = []
    profit_factor_std = []
    profit_factor_cv = []

    from_time_list = []
    to_time_list = []
    positive = 0
    negative = 0
    from_time_timestamps = []
    pnl_percent_pos = []

    # day_seconds = days * 24 * 60 * 60
    # hour_seconds = hours * 60 * 60
    total_seconds = time_delta

    i = 0
    while i < tests:
        if pool == "last year":
            db = Hdf5Client()
            oldest_ts, most_recent_ts = db.get_first_last_timestamp(contract)
            from_time, to_time = random_start_end(contract, tf, total_seconds, "l")
        else:
            from_time, to_time = random_start_end(contract, tf, total_seconds, "f")
        for j in from_time_timestamps:
            while True:
                if abs(from_time - j) < 86400:
                    if pool == "last year":
                        from_time, to_time = random_start_end(contract, tf, total_seconds, "l")
                    else:
                        from_time, to_time = random_start_end(contract, tf, total_seconds, "f")
                else:
                    break
        if strategy == "guppy":
            h5_db = Hdf5Client()
            data = h5_db.get_data(contract, from_time, to_time)
            data = resample_timeframe(data, tf)
            pnl, max_drawdown, win_rate, rr_long, rr_short, num_trades, mod_win_rate, max_losses, max_wins, \
                trades_won, trades_lost, breakeven_trades, profit_factor, num_longs, num_shorts \
                = strategies.guppy.backtest(df=data, initial_capital=initial_capital,
                                            trade_longs=params['trade_longs'],
                                            trade_shorts=params['trade_shorts'], sl_long=params['sl_long'],
                                            sl_short=params['sl_short'], mfi_long=params['mfi_long'],
                                            mfi_short=params['mfi_short'], mfi_period=params['mfi_period'],
                                            mfi_mult=params['mfi_mult'], mfi_ypos=params['mfi_ypos'],
                                            mfi_long_threshold=params['mfi_long_threshold'],
                                            mfi_short_threshold=params['mfi_short_threshold'],
                                            macd_short=params['macd_short'], macd_fast=params['macd_fast'],
                                            macd_slow=params['macd_slow'], macd_signal=params['macd_signal'],
                                            macd_long=params['macd_long'], rsi_long=params['rsi_long'],
                                            rsi_short=params['rsi_short'], rsi_length=params['rsi_length'],
                                            rsi_overbought=params['rsi_overbought'],
                                            rsi_oversold=params['rsi_oversold'],
                                            ema200_long=params['ema200_long'],
                                            ema200_short=params['ema200_short'],
                                            guppy_fast_long=params['guppy_fast_long'],
                                            guppy_fast_short=params['guppy_fast_short'],
                                            ribbon_check_long=params['ribbon_check_long'],
                                            ribbon_check_short=params['ribbon_check_short'],
                                            move_sl_type_long=params['move_sl_type_long'],
                                            move_sl_type_short=params['move_sl_type_short'],
                                            move_sl_long=params['move_sl_long'],
                                            move_sl_short=params['move_sl_short'], risk=params['risk'],
                                            leverage=params['leverage'], tp_long=params['tp_long'],
                                            tp_short=params['tp_short'], ltp1=params['ltp1'],
                                            ltp1_qty=params['ltp1_qty'], ltp2=params['ltp2'],
                                            ltp2_qty=params['ltp2_qty'], ltp3=params['ltp3'],
                                            stp1=params['stp1'], stp1_qty=params['stp1_qty'],
                                            stp2=params['stp2'], stp2_qty=params['stp2_qty'],
                                            stp3=params['stp3'], mode="m", contract=contract, tf=tf,
                                            from_time=from_time, to_time=to_time,
                                            bb_long=params['bb_long'], bb_short=params['bb_short'],
                                            bb_length=params['bb_length'],
                                            bb_mult=params['bb_mult'],
                                            wae_long=params['wae_long'], wae_short=params['wae_short'],
                                            wae_sensitivity=params['wae_sensitivity'],
                                            wae_fast_length=params['wae_fast_length'],
                                            wae_slow_length=params['wae_slow_length'],
                                            wae_bb_length=params['wae_bb_length'],
                                            wae_bb_mult=params['wae_bb_mult'],
                                            wae_rma_length=params['wae_rma_length'],
                                            wae_dz_mult=params['wae_dz_mult'],
                                            wae_expl_check=params['wae_expl_check'],
                                            adx_long=params['adx_long'], adx_short=params['adx_short'],
                                            adx_smoothing=params['adx_smoothing'],
                                            adx_di_length=params['adx_di_length'],
                                            adx_length_long=params['adx_length_long'],
                                            adx_length_short=params['adx_length_short'],
                                            )

            if pnl > 0:
                positive += 1
            else:
                negative += 1
            pnl_percent_pos.append((positive / (i + 1)) * 100)
            pnl_list.append(pnl)
            pnl_avg.append(np.mean(pnl_list))
            pnl_std.append(np.std(pnl_list))
            if np.mean(pnl_list) != 0:
                pnl_cv.append(np.std(pnl_list)/np.mean(pnl_list))
            else:
                pnl_cv.append(np.NaN)

            max_dd_list.append(max_drawdown)
            max_dd_avg.append(np.mean(max_dd_list))
            max_dd_std.append(np.std(max_dd_list))
            if np.mean(max_dd_list) != 0:
                max_dd_cv.append(np.std(max_dd_list)/np.mean(max_dd_list))
            else:
                max_dd_cv.append(np.NaN)

            win_rate_list.append(win_rate)
            win_rate_avg.append(np.mean(win_rate_list))
            win_rate_std.append(np.std(win_rate_list))
            if np.mean(win_rate_list) != 0:
                win_rate_cv.append(np.std(win_rate_list)/np.mean(win_rate_list))
            else:
                win_rate_cv.append(np.NaN)

            mod_win_rate_list.append(mod_win_rate)
            mod_win_rate_avg.append(np.mean(mod_win_rate_list))
            mod_win_rate_std.append(np.std(mod_win_rate_list))
            if np.mean(mod_win_rate_list) != 0:
                mod_win_rate_cv.append(np.std(mod_win_rate_list)/np.mean(mod_win_rate_list))
            else:
                mod_win_rate_cv.append(np.NaN)

            num_trades_list.append(num_trades)
            num_trades_avg.append(np.mean(num_trades_list))
            num_trades_std.append(np.std(num_trades_list))
            if np.mean(num_trades_list) != 0:
                num_trades_cv.append(np.std(num_trades_list)/np.mean(num_trades_list))
            else:
                num_trades_cv.append(np.NaN)

            num_longs_list.append(num_longs)
            num_longs_avg.append(np.mean(num_longs_list))
            num_longs_std.append(np.std(num_longs_list))
            if np.mean(num_longs_list) != 0:
                num_longs_cv.append(np.std(num_longs_list) / np.mean(num_longs_list))
            else:
                num_longs_cv.append(np.NaN)

            num_shorts_list.append(num_shorts)
            num_shorts_avg.append(np.mean(num_shorts_list))
            num_shorts_std.append(np.std(num_shorts_list))
            if np.mean(num_shorts_list) != 0:
                num_shorts_cv.append(np.std(num_shorts_list) / np.mean(num_shorts_list))
            else:
                num_shorts_cv.append(np.NaN)

            max_losses_list.append(max_losses)
            max_losses_avg.append(np.mean(max_losses_list))
            max_losses_std.append(np.std(max_losses_list))
            if np.mean(max_losses_list) != 0:
                max_losses_cv.append(np.std(max_losses_list)/np.mean(max_losses_list))
            else:
                max_losses_cv.append(np.NaN)

            max_wins_list.append(max_wins)
            max_wins_avg.append(np.mean(max_wins_list))
            max_wins_std.append(np.std(max_wins_list))
            if np.mean(max_wins_list) != 0:
                max_wins_cv.append(np.std(max_wins_list)/np.mean(max_wins_list))
            else:
                max_wins_cv.append(np.NaN)

            trades_won_list.append(trades_won)
            trades_won_avg.append(np.mean(trades_won_list))
            trades_won_std.append(np.std(trades_won_list))
            if np.mean(trades_won_list) != 0:
                trades_won_cv.append(np.std(trades_won_list) / np.mean(trades_won_list))
            else:
                trades_won_cv.append(np.NaN)

            trades_lost_list.append(trades_lost)
            trades_lost_avg.append(np.mean(trades_lost_list))
            trades_lost_std.append(np.std(trades_lost_list))
            if np.mean(trades_lost_list) != 0:
                trades_lost_cv.append(np.std(trades_lost_list) / np.mean(trades_lost_list))
            else:
                trades_lost_cv.append(np.NaN)

            breakeven_trades_list.append(breakeven_trades)
            breakeven_trades_avg.append(np.mean(breakeven_trades_list))
            breakeven_trades_std.append(np.std(breakeven_trades_list))
            if np.mean(breakeven_trades_list) != 0:
                breakeven_trades_cv.append(np.std(breakeven_trades_list) / np.mean(breakeven_trades_list))
            else:
                breakeven_trades_cv.append(np.NaN)

            profit_factor_list.append(profit_factor)
            profit_factor_avg.append(np.mean(profit_factor_list))
            profit_factor_std.append(np.std(profit_factor_list))
            if np.mean(profit_factor_list) != 0:
                profit_factor_cv.append(np.std(profit_factor_list) / np.mean(profit_factor_list))
            else:
                profit_factor_cv.append(np.NaN)

            from_time_timestamps.append(from_time + (60 * TF_SECONDS[tf]))
            start = datetime.datetime.fromtimestamp(from_time + (60 * TF_SECONDS[tf]))
            start = start.strftime("%Y-%m-%d-%I:%M%p")
            end = datetime.datetime.fromtimestamp(to_time)
            end = end.strftime("%Y-%m-%d-%I:%M%p")
            from_time_list.append(start)
            to_time_list.append(end)
            # return pnl, max_drawdown, win_rate, rr_long, rr_short, num_trades, max_losses, max_wins
            if mode == "s":
                print(f"\rTest {i+1} of {tests}", end=" ")
            else:
                print(f"\rParameter set #{iteration+1} Test {i+1} of {tests}", end=" ")

        i += 1

    # print("\n")

    df['from_time'] = from_time_list
    df['to_time'] = to_time_list

    df['pnl'] = pnl_list
    df['pnl_avg'] = pnl_avg
    df['pnl_std'] = pnl_std
    df['pnl_cv'] = pnl_cv
    df['%_positive'] = pnl_percent_pos

    df['max_dd'] = max_dd_list
    df['max_dd_avg'] = max_dd_avg
    df['max_dd_std'] = max_dd_std
    df['max_dd_cv'] = max_dd_cv

    df['win_rate'] = win_rate_list
    df['win_rate_avg'] = win_rate_avg
    df['win_rate_std'] = win_rate_std
    df['win_rate_cv'] = win_rate_cv

    df['mod_win_rate'] = mod_win_rate_list
    df['mod_win_rate_avg'] = mod_win_rate_avg
    df['mod_win_rate_std'] = mod_win_rate_std
    df['mod_win_rate_cv'] = mod_win_rate_cv

    df['num_trades'] = num_trades_list
    df['num_trades_avg'] = num_trades_avg
    df['num_trades_std'] = num_trades_std
    df['num_trades_cv'] = num_trades_cv

    df['num_longs'] = num_longs_list
    df['num_longs_avg'] = num_longs_avg
    df['num_longs_std'] = num_longs_std
    df['num_longs_cv'] = num_longs_cv

    df['num_shorts'] = num_shorts_list
    df['num_shorts_avg'] = num_shorts_avg
    df['num_shorts_std'] = num_shorts_std
    df['num_shorts_cv'] = num_shorts_cv

    df['trades_won'] = trades_won_list
    df['trades_won_avg'] = trades_won_avg
    df['trades_won_std'] = trades_won_std
    df['trades_won_cv'] = trades_won_cv

    df['trades_lost'] = trades_lost_list
    df['trades_lost_avg'] = trades_lost_avg
    df['trades_lost_std'] = trades_lost_std
    df['trades_lost_cv'] = trades_lost_cv

    df['breakeven_trades'] = breakeven_trades_list
    df['breakeven_trades_avg'] = breakeven_trades_avg
    df['breakeven_trades_std'] = breakeven_trades_std
    df['breakeven_trades_cv'] = breakeven_trades_cv

    df['max_wins'] = max_wins_list
    df['max_wins_avg'] = max_wins_avg
    df['max_wins_std'] = max_wins_std
    df['max_wins_cv'] = max_wins_cv

    df['max_losses'] = max_losses_list
    df['max_losses_avg'] = max_losses_avg
    df['max_losses_std'] = max_losses_std
    df['max_losses_cv'] = max_losses_cv

    df['profit_factor'] = profit_factor_list
    df['profit_factor_avg'] = profit_factor_avg
    df['profit_factor_std'] = profit_factor_std
    df['profit_factor_cv'] = profit_factor_cv

    df['rr_long'] = rr_long
    df['rr_short'] = rr_short
    df.index += 1

    if mode == "s":
        csv_data.reset_index(inplace=True)
        df = pd.concat([df, csv_data], axis=0)

        if os.path.exists(f"Results/Multitests/MultitestResults_{file_name}_{tests}_tests.csv"):
            while True:
                try:
                    myfile = open(f"Results/Multitests/MultitestResults_{file_name}_{tests}_"
                                  f"tests.csv", "w+")
                    break
                except IOError:
                    input(f"Cannot write results to csv file. Please close \n"
                          f"Results/Multitests/MultitestResults_{file_name}_{tests}_"
                          f"tests.csv\nThen press Enter to "
                          f"retry.")
        df.to_csv(f"Results/Multitests/MultitestResults_{file_name}_{tests}_tests.csv")
        return df
    else:
        return df


def mega_futuretest(contract: Contract, strategy: str, tf: str, initial_capital: int, file_name: str,
                    min_per_delta: int, op_df: pd.DataFrame) -> pd.DataFrame:
    param_descriptions = STRAT_PARAMS[strategy]
    params = dict()
    df = pd.DataFrame()
    # while True:
    #
    #     if file_exists(file_name):
    #         break
    #     else:
    #         print(f"ERROR: {file_name} does not exist")
    #         continue

    csv_data = pd.read_csv(file_name, header=0)
    csv_data = csv_data.iloc[:, 1:]
    rows = csv_data.index[-1] + 1
    file_name = file_name.rstrip(".csv")
    version, c, dates = file_name.partition(f"OptimizerResults_{contract.symbol}_{tf}_")
    version = version.lstrip("Results/Optimizations/")
    start_date, a, end_date = dates.partition("_to_")
    end_date, a, b = end_date.partition("_")
    start_time = start_date
    end_time = end_date
    start_date = int(datetime.datetime.strptime(start_date, "%Y-%m-%d-%I%p").timestamp())
    start_date -= (60 * TF_SECONDS[tf])
    end_date = int(datetime.datetime.strptime(end_date, "%Y-%m-%d-%I%p").timestamp())
    pnl_list = []
    pnl_delta = []
    pnl_percent_delta = []
    max_dd_delta = []
    win_rate_delta = []
    rr_long_list = []
    rr_short_list = []
    num_trades_delta = []
    num_longs_delta = []
    num_shorts_delta = []
    mod_win_rate_delta = []
    max_losses_delta = []
    max_wins_delta = []

    iterations = 0
    while iterations < rows:
        for p_code, p in param_descriptions.items():
            if p["type"] == Y_N:
                params[p_code] = str(csv_data.at[iterations, p_code])
            else:
                params[p_code] = p["type"](csv_data.at[iterations, p_code])
        fpnl, fmax_drawdown, fwin_rate, rr_long, rr_short, fnum_trades, fmod_win_rate, fmax_losses, fmax_wins, \
            fnum_longs, fnum_shorts \
            = f_tester(contract, strategy, tf, start_date, initial_capital, params, csv_data, "m", iterations)
        pnl_list.append(fpnl)
        pnl_delta.append(fpnl - float(csv_data.at[iterations, "pnl"]))
        try:
            pnl_percent_delta.append((((fpnl + initial_capital) / (float(csv_data.at[iterations, "pnl"]) + 100)) - 1)
                                     * 100)
        except ZeroDivisionError:
            pnl_percent_delta.append(0)
        max_dd_delta.append(fmax_drawdown - float(csv_data.at[iterations, "max_dd"]))
        win_rate_delta.append(fwin_rate - float(csv_data.at[iterations, "win_rate"]))
        rr_long_list.append(rr_long)
        rr_short_list.append(rr_short)
        if fnum_trades != 0:
            num_trades_delta.append(fnum_trades - int(csv_data.at[iterations, "num_trades"]))
            num_longs_delta.append(fnum_longs - int(csv_data.at[iterations, "num_longs"]))
            num_shorts_delta.append(fnum_shorts - int(csv_data.at[iterations, "num_shorts"]))
        else:
            num_trades_delta.append(0)
            num_longs_delta.append(0)
            num_shorts_delta.append(0)
        mod_win_rate_delta.append(fmod_win_rate - float(csv_data.at[iterations, "mod_win_rate"]))
        max_losses_delta.append(fmax_losses - int(csv_data.at[iterations, "max_losses"]))
        max_wins_delta.append(fmax_wins - int(csv_data.at[iterations, "max_wins"]))
        print(f"\rParameter set #{iterations+1} of {rows}", end=" ")
        iterations += 1
    print("\n")

    df["new_pnl"] = pnl_list
    df["pnl_delta"] = pnl_delta
    df["pnl_%_delta"] = pnl_percent_delta
    df["max_dd_delta"] = max_dd_delta
    df["win_rate_delta"] = win_rate_delta
    df["mod_win_rate_delta"] = mod_win_rate_delta
    df["rr_long"] = rr_long_list
    df["rr_short"] = rr_short_list
    df["num_trades_delta"] = num_trades_delta
    df["num_longs_delta"] = num_longs_delta
    df["num_shorts_delta"] = num_shorts_delta
    df["max_losses_delta"] = max_losses_delta
    df["max_wins_delta"] = max_wins_delta
    df["original_pnl"] = csv_data["pnl"]
    df["original_max_dd"] = csv_data["max_dd"]
    df["original_win_rate"] = csv_data["win_rate"]
    df["original_mod_win_rate"] = csv_data["mod_win_rate"]
    df["original_num_trades"] = csv_data["num_trades"]
    df["original_num_longs"] = csv_data["num_longs"]
    df["original_num_shorts"] = csv_data["num_shorts"]
    df["original_max_losses"] = csv_data["max_losses"]
    df["original_max_wins"] = csv_data["max_wins"]
    # df.index += 2

    for i in range(len(pnl_percent_delta)):
        if pnl_percent_delta[i] < min_per_delta:
            df = df.drop(index=i)
            op_df = op_df.drop(index=i)

    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%I%p")

    if os.path.exists(f"Results/FutureTests/{version} MegaFutureTest_{contract.symbol}_{tf}_{start_time}_to"
                      f"_{end_time}_to"
                      f"_{current_time}.csv"):
        while True:
            try:
                myfile = open(f"Results/FutureTests/{version} MegaFutureTest_{contract.symbol}_{tf}_{start_time}_to"
                              f"_{end_time}_to"
                                f"_{current_time}.csv",
                              "w+")
                break
            except IOError:
                input(f"Cannot write results to csv file. Please close \n"
                      f"Results/FutureTests/{version} MegaFutureTest_{contract.symbol}_{tf}_{start_time}_to"
                      f"_{end_time}_to"
                      f"_{current_time}.csv"
                      f"\nThen press Enter to "
                      f"retry.")
    df.to_csv(f"Results/FutureTests/{version} MegaFutureTest_{contract.symbol}_{tf}_{start_time}_to_{end_time}_to"
              f"_{current_time}.csv")
    return op_df


def f_tester(contract: Contract, strategy: str, tf: str, start_time: int, initial_capital: int,
                 params: typing.Dict, csv_data: pd.DataFrame, mode: str, iteration: int):
    if strategy == "guppy":
        h5_db = Hdf5Client()
        data = h5_db.get_data(contract, start_time, int(time.time()))
        data = resample_timeframe(data, tf)
        pnl, max_drawdown, win_rate, rr_long, rr_short, trade_counter, mod_win_rate, max_losses, max_wins, \
            trades_won, trades_lost, breakeven_trades, profit_factor, num_longs, num_shorts \
            = strategies.guppy.backtest(df=data, initial_capital=initial_capital,
                                        trade_longs=params['trade_longs'],
                                        trade_shorts=params['trade_shorts'], sl_long=params['sl_long'],
                                        sl_short=params['sl_short'], mfi_long=params['mfi_long'],
                                        mfi_short=params['mfi_short'], mfi_period=params['mfi_period'],
                                        mfi_mult=params['mfi_mult'], mfi_ypos=params['mfi_ypos'],
                                        mfi_long_threshold=params['mfi_long_threshold'],
                                        mfi_short_threshold=params['mfi_short_threshold'],
                                        adx_long=params['adx_long'], adx_short=params['adx_short'],
                                        macd_short=params['macd_short'], macd_fast=params['macd_fast'],
                                        macd_slow=params['macd_slow'], macd_signal=params['macd_signal'],
                                        macd_long=params['macd_long'], rsi_long=params['rsi_long'],
                                        rsi_short=params['rsi_short'], rsi_length=params['rsi_length'],
                                        rsi_overbought=params['rsi_overbought'], rsi_oversold=params['rsi_oversold'],
                                        ema200_long=params['ema200_long'],
                                        ema200_short=params['ema200_short'],
                                        guppy_fast_long=params['guppy_fast_long'],
                                        guppy_fast_short=params['guppy_fast_short'],
                                        ribbon_check_long=params['ribbon_check_long'],
                                        ribbon_check_short=params['ribbon_check_short'],
                                        move_sl_type_long=params['move_sl_type_long'],
                                        move_sl_type_short=params['move_sl_type_short'],
                                        move_sl_long=params['move_sl_long'],
                                        move_sl_short=params['move_sl_short'], risk=params['risk'],
                                        leverage=params['leverage'], tp_long=params['tp_long'],
                                        tp_short=params['tp_short'], ltp1=params['ltp1'],
                                        ltp1_qty=params['ltp1_qty'], ltp2=params['ltp2'],
                                        ltp2_qty=params['ltp2_qty'], ltp3=params['ltp3'],
                                        stp1=params['stp1'], stp1_qty=params['stp1_qty'],
                                        stp2=params['stp2'], stp2_qty=params['stp2_qty'],
                                        stp3=params['stp3'], mode=mode, contract=contract, tf=tf,
                                        from_time=start_time, to_time=int(time.time()),
                                        bb_long=params['bb_long'], bb_short=params['bb_short'],
                                        bb_length=params['bb_length'],
                                        bb_mult=params['bb_mult'],
                                        wae_long=params['wae_long'], wae_short=params['wae_short'],
                                        wae_sensitivity=params['wae_sensitivity'],
                                        wae_fast_length=params['wae_fast_length'],
                                        wae_slow_length=params['wae_slow_length'],
                                        wae_bb_length=params['wae_bb_length'],
                                        wae_bb_mult=params['wae_bb_mult'],
                                        wae_rma_length=params['wae_rma_length'],
                                        wae_dz_mult=params['wae_dz_mult'],
                                        wae_expl_check=params['wae_expl_check'],
                                        adx_smoothing=params['adx_smoothing'],
                                        adx_di_length=params['adx_di_length'],
                                        adx_length_long=params['adx_length_long'],
                                        adx_length_short=params['adx_length_short'], )

        return pnl, max_drawdown, win_rate, rr_long, rr_short, trade_counter, mod_win_rate, max_losses, max_wins, \
            num_longs, num_shorts
