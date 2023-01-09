from bybit import BybitClient
from data_collector import collect_all
from utils import *
import logging.handlers
import time
import datetime
import backtester
import optimizer
import pandas as pd
import os.path
import smtplib
import numpy as np
from configparser import ConfigParser
from os.path import exists as file_exists

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(levelname)s :: %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler("info.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
# logger.addHandler(smtp_handler)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

parser = ConfigParser()
parser.read('config.ini')
email = parser.get('email', 'email address')


if __name__ == "__main__":
    client = BybitClient()
    modes = ["data", "backtest", "speedtest", "stnop"]
    while True:
        mode = input("Choose the program mode (data / speedtest / speedtest no op (stnop) / backtest): ").lower()
        if mode in modes:
            break
    contracts = []
    for a in client.contracts:
        contracts.append(client.contracts[a].symbol)

    while True:
        if (mode == "backtest") or (mode == "speedtest"):
            symbol = input("Choose an asset pair: ").upper()
            if symbol in contracts:
                break
        else:
            symbol = input("Choose an asset pair or Press Enter for all: ").upper()
            if symbol in contracts:
                break
            elif symbol == "":
                break

    for i in client.contracts:
        if client.contracts[i].symbol == symbol:
            symbol = client.contracts[i]

    if mode == "data":
        start_time = time.time()
        if symbol != "":
            collect_all(client, symbol, 1622413603)
        else:
            for pair in client.usdt_contracts:
                collect_all(client, client.contracts[pair], 1622413603)

    elif mode in ["backtest", "speedtest", "stnop"]:

        # Strategy
        # available_strategies = ["obv", "ichimoku", "sup_res", "mfi", "guppy"]
        # while True:
        #     strategy = input(f"Choose a strategy ({', '.join(available_strategies)}): ").lower()
        #     if strategy in available_strategies:
        #         break
        strategy = "guppy"

        # Timeframe
        while True:
            timeframe = input(f"Choose a timeframe ({', '.join(TF_EQUIV.keys())}): ").lower()
            if timeframe in TF_EQUIV.keys():
                break

        # From
        while True:
            from_time = input("Backtest from (yyyy-mm-dd hh) or Press Enter: ")
            if from_time == "":
                from_time = 0
                break
            try:
                from_time = int(datetime.datetime.strptime(from_time, "%Y-%m-%d %H").timestamp())
                break
            except ValueError:
                continue
        from_time -= (60 * TF_SECONDS[timeframe])

        # To
        while True:
            to_time = input("Backtest until (yyyy-mm-dd hh) or Press Enter for current time: ")
            if to_time == "":
                to_time = time.time()
                break
            try:
                to_time = int(datetime.datetime.strptime(to_time, "%Y-%m-%d %H").timestamp())
                break
            except ValueError:
                continue

        # Initial Capital
        while True:
            initial_capital = input("Initial capital to trade with or Press Enter for 100: ")
            if initial_capital == "":
                initial_capital = int(100)
                break
            else:
                try:
                    initial_capital = int(initial_capital)
                    break
                except ValueError:
                    continue

        if mode == "backtest":
            start_time = time.time()
            while True:
                file_name = str(input("Input CSV file name to read parameters from (including .csv): "))

                if file_exists(file_name):
                    break
                else:
                    print(f"ERROR: {file_name} does not exist")
                    continue
            backtester.run(symbol, strategy, timeframe, from_time, to_time, initial_capital, file_name, "backtest")

        elif mode == "speedtest":
            # Population size
            while True:
                try:
                    pop_size = int(input(f"Choose a population size: "))
                    break
                except ValueError:
                    continue

            # Iterations (generations)
            while True:
                try:
                    generations = int(input(f"Choose a number of generations: "))
                    break
                except ValueError:
                    continue
            tests = 0

            # Min pnl % delta to keep results
            while True:
                try:
                    min_per_delta = int(input(f"Minimum PNL % delta to keep future test results: "))
                    break
                except ValueError:
                    continue

            start_time = time.time()
            nsga2 = optimizer.Nsga2(symbol, strategy, timeframe, from_time, to_time, initial_capital, pop_size)
            p_population = nsga2.create_initial_population()
            p_population = nsga2.evaluate_population(p_population)
            if tests != 0:
                d1 = nsga2.rss_period()
                rss_population = nsga2.rss_backtest(d1, p_population)
                p_population = nsga2.cv_calculation(p_population, rss_population)
            p_population = nsga2.crowding_distance(p_population)

            g = 0

            while g < generations:
                # Create offspring
                q_population = nsga2.create_offspring_population(p_population)
                # Evaluate
                q_population = nsga2.evaluate_population(q_population)
                r_population = p_population + q_population

                nsga2.population_params.clear()

                i = 0
                population = dict()
                for bt in r_population:
                    bt.reset_results()
                    nsga2.population_params.append(bt.parameters)
                    population[i] = bt
                    i += 1

                # RSS tests
                k = 0
                while k < tests:
                    d1 = nsga2.rss_period()
                    rss_population = nsga2.rss_backtest(d1, population)
                    print(f"\r"
                          f"{int(((g/generations)*100+((k+1)/(generations*tests)*100)))}% "
                          f" Gen {(g + 1)} Test {k + 1}, RSS period: {d1.index[0]} to {d1.index[-1]} ", end=" ")
                    k += 1

                if tests != 0:
                    population = nsga2.cv_calculation(population, rss_population)
                fronts = nsga2.non_dominated_sorting(population)

                for j in range(len(fronts)):
                    fronts[j] = nsga2.crowding_distance((fronts[j]))

                p_population = nsga2.create_new_population(fronts)

                if tests == 0:
                    print(f"\r{int((g + 1) / generations * 100)}%", end=" ")

                g += 1

            print("\n")

            df = pd.DataFrame()

            for individual in p_population:
                print(individual)
                ps = pd.Series(individual.parameters)
                ps = pd.DataFrame(ps)
                ps = ps.transpose()
                ps["pnl"] = individual.pnl
                ps["pnl_avg"] = np.mean(individual.pnl_history)
                ps["pnl_std"] = np.std(individual.pnl_history)
                ps["pnl_cv"] = individual.pnl_cv
                ps["max_dd"] = individual.max_dd
                ps["max_dd_avg"] = np.mean(individual.max_dd_history)
                ps["max_dd_std"] = np.std(individual.max_dd_history)
                ps["max_dd_cv"] = individual.max_dd_cv
                ps["pnl_dd_ratio"] = individual.pnl_dd_ratio
                ps["pnl_dd_ratio_cv"] = individual.pnl_dd_ratio_cv
                ps["win_rate"] = individual.win_rate
                ps["win_rate_cv"] = individual.win_rate_cv
                ps["mod_win_rate"] = individual.mod_win_rate
                ps["risk_reward_long"] = individual.rr_long
                ps["risk_reward_short"] = individual.rr_short
                ps["num_trades"] = individual.num_trades
                ps["num_longs"] = individual.num_longs
                ps["num_shorts"] = individual.num_shorts
                ps["max_losses"] = individual.max_losses
                ps["max_wins"] = individual.max_wins
                df = pd.concat([df, ps], axis=0, ignore_index=True)

            start = datetime.datetime.fromtimestamp(from_time + (60 * TF_SECONDS[timeframe]))
            start = start.strftime("%Y-%m-%d-%I%p")
            end = datetime.datetime.fromtimestamp(to_time)
            end = end.strftime("%Y-%m-%d-%I%p")

            file_name = f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_{start}_to_{end}" \
                        f"_{pop_size}x" \
                        f"{generations}x{tests}.csv"
            if os.path.exists(f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_"
                              f"{start}_to_{end}_{pop_size}x{generations}x{tests}.csv"):
                while True:
                    try:
                        myfile = open(f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_"
                                      f"{start}_to_{end}_{pop_size}x{generations}x{tests}.csv", "w+")
                        break
                    except IOError:
                        input(f"Cannot write results to csv file. Please close \n"
                              f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_"
                              f"{start}_to_{end}_{pop_size}x{generations}x{tests}.csv\nThen press Enter to "
                              f"retry.")
            df.to_csv(f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_{start}_to_"
                      f"{end}_{pop_size}x{generations}x{tests}.csv")

            collect_all(client, symbol, 1622413603)
            df = backtester.mega_futuretest(symbol, strategy, timeframe, initial_capital, file_name, min_per_delta,
                                            df, "speedtest")

            if os.path.exists(f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_"
                              f"{start}_to_{end}_{pop_size}x{generations}x{tests}.csv"):
                while True:
                    try:
                        myfile = open(f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_"
                                      f"{start}_to_{end}_{pop_size}x{generations}x{tests}.csv", "w+")
                        break
                    except IOError:
                        input(f"Cannot write results to csv file. Please close \n"
                              f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_"
                              f"{start}_to_{end}_{pop_size}x{generations}x{tests}.csv\nThen press Enter to "
                              f"retry.")
            df.to_csv(f"Results/Optimizations/OptimizerResults_{symbol.symbol}_{timeframe}_{start}_to_"
                      f"{end}_{pop_size}x{generations}x{tests}.csv")

            if len(df.index) >= 1:
                num_results = len(df.index)
                multitest_type = "mp"
                pool_type = "last year"
                time_delta = to_time - from_time
                mtests = 1
                backtester.multitest(symbol, strategy, timeframe, multitest_type, time_delta, initial_capital, mtests,
                                     pool_type, file_name)

                backtester.run(symbol, strategy, timeframe, from_time, int(time.time()), initial_capital, file_name,
                               "speedtest")

                try:
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. {num_results} results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()
                except smtplib.SMTPException as e:
                    logger.error(e)
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. {num_results} results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()
            else:
                print("No results")
                try:
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report - No Results"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. No results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()
                except smtplib.SMTPException as e:
                    logger.error(e)
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report - No Results"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. No results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()

        elif mode == "stnop":
            start_time = time.time()
            strategy = "guppy"
            # Population size
            while True:
                try:
                    pop_size = int(input(f"Choose a population size: "))
                    break
                except ValueError:
                    continue

            # Iterations (generations)
            while True:
                try:
                    generations = int(input(f"Choose a number of generations: "))
                    break
                except ValueError:
                    continue
            while True:
                file_name = str(input("Input CSV file name to read parameters from (including .csv): "))

                if file_exists(file_name):
                    break
                else:
                    print(f"ERROR: {file_name} does not exist")
                    continue
            file_data = file_name.lstrip(f"OptimizerResults_{symbol}_")
            file_data = file_data.rstrip(".csv")

            df = pd.read_csv(file_name, header=0)
            df = df.iloc[:, 1:]
            rows = df.index[-1] + 1
            multitest_results = pd.DataFrame()

            # Min pnl % delta to keep results
            while True:
                try:
                    min_per_delta = int(input(f"Minimum PNL % delta to keep future test results: "))
                    break
                except ValueError:
                    continue

            collect_all(client, symbol, 1622413603)
            df = backtester.mega_futuretest(symbol, strategy, timeframe, initial_capital, file_name, min_per_delta,
                                            df, "stnop")
            start = datetime.datetime.fromtimestamp(from_time + (60 * TF_SECONDS[timeframe]))
            start = start.strftime("%Y-%m-%d-%I%p")
            end = datetime.datetime.fromtimestamp(to_time)
            end = end.strftime("%Y-%m-%d-%I%p")
            tests = 0
            if len(df.index) >= 1:
                num_results = len(df.index)
                multitest_type = "mp"
                pool_type = "last year"
                time_delta = to_time - from_time
                mtests = 1
                backtester.multitest(symbol, strategy, timeframe, multitest_type, time_delta, initial_capital, mtests,
                                     pool_type, file_name)

                backtester.run(symbol, strategy, timeframe, from_time, int(time.time()), initial_capital, file_name,
                               "speedtest")

                try:
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. {num_results} results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()
                except smtplib.SMTPException as e:
                    logger.error(e)
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. {num_results} results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()
            else:
                print("No results")
                try:
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report - No Results"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. No results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()
                except smtplib.SMTPException as e:
                    logger.error(e)
                    smtp = smtplib.SMTP("smtp.gmail.com", 587)
                    smtp.starttls()
                    smtp.login("guppy.bot.messenger@gmail.com", "otqcxemvnbxvfjpe")
                    subject = "Speed BTOM Report - No Results"
                    text = f"{pop_size}x{generations}x{tests} run for {symbol.symbol}_{timeframe} from " \
                           f"{start} to {end} complete. No results found."
                    message = "Subject: {} \n\n {}".format(subject, text)
                    smtp.sendmail("guppy.bot.messenger@gmail.com", email, message)
                    smtp.quit()

# print(f"Time to complete: {int(time.time() - start_time)} seconds")

a = input("Press Enter to close.")
