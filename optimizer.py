import datetime

import numpy as np

from utils import *
from database import Hdf5Client
from models import *
import random
import typing
import copy

import strategies.obv
import strategies.ichimoku
import strategies.support_resistance
import strategies.mfi
import strategies.guppy

import logging
logger = logging.getLogger()


class Nsga2:
    def __init__(self, contract: Contract, strategy: str, tf: str, from_time: int, to_time: int,
                 initial_capital: int, population_size: int):
        self.contract = contract
        self.symbol = contract.symbol
        self.strategy = strategy
        self.tf = tf
        self.from_time = from_time
        self.to_time = to_time
        self.population_size = population_size
        self.initial_capital = initial_capital

        self.params_data = STRAT_PARAMS[strategy]

        self.population_params = []

        if self.strategy in ["obv", "ichimoku", "sup_res", "guppy"]:
            h5_db = Hdf5Client()
            self.data = h5_db.get_data(contract, from_time, to_time)
            self.data = resample_timeframe(self.data, tf)

        # elif self.strategy in ["psar", "sma"]:
        #     self.lib = get_library()
        #
        #     if self.strategy == "sma":
        #         self.obj = self.lib.Sma_new(exchange.encode(), symbol.encode(), tf.encode(), from_time, to_time)

    def create_initial_population(self) -> typing.List[BacktestResult]:
        population = []
        i = 1

        while len(population) < self.population_size:
            backtest = BacktestResult()
            for p_code, p in self.params_data.items():
                if p["type"] == int:
                    backtest.parameters[p_code] = random.randint(p["min"], p["max"])
                elif p["type"] == float:
                    backtest.parameters[p_code] = round(random.uniform(p["min"], p["max"]), p["decimals"])
                elif p["type"] == Y_N:
                    backtest.parameters[p_code] = random.choice(p["choices"])
                    # if result == 0:
                    #     backtest.parameters[p_code] = "N"
                    # else:
                    #     backtest.parameters[p_code] = "Y"

            backtest.parameters = self._params_constraints(backtest.parameters)
            # pprint.pprint(backtest.parameters)
            # with open(f'parameters_{i}.csv', 'w') as f:
            #     for key in backtest.parameters.keys():
            #         f.write("%s, %s\n" % (key, backtest.parameters[key]))

            if backtest not in population:
                population.append(backtest)
                self.population_params.append(backtest.parameters)
            i += 1

        return population

    def create_new_population(self, fronts: typing.List[typing.List[BacktestResult]]) -> typing.List[BacktestResult]:

        new_pop = []

        for front in fronts:
            if len(new_pop) + len(front) > self.population_size:
                max_individuals = self.population_size - len(new_pop)
                if max_individuals > 0:
                    new_pop += sorted(front, key=lambda x: getattr(x, "crowding_distance"))[-max_individuals:]
            else:
                new_pop += front

        return new_pop

    def create_offspring_population(self, population: typing.List[BacktestResult]) -> typing.List[BacktestResult]:

        offspring_pop = []

        while len(offspring_pop) != self.population_size:

            parents: typing.List[BacktestResult] = []

            for i in range(2):
                random_parents = random.sample(population, k=2)
                if random_parents[0].rank != random_parents[1].rank:
                    best_parent = min(random_parents, key=lambda x: getattr(x, "rank"))
                else:
                    best_parent = max(random_parents, key=lambda x: getattr(x, "crowding_distance"))

                parents.append(best_parent)

            new_child = BacktestResult()
            new_child.parameters = copy.copy(parents[0].parameters)

            # Crossover

            number_of_crossovers = random.randint(1, len(self.params_data))
            params_to_cross = random.sample(list(self.params_data.keys()), k=number_of_crossovers)

            for p in params_to_cross:
                new_child.parameters[p] = copy.copy(parents[1].parameters[p])

            # Mutation

            number_of_mutations = random.randint(0, len(self.params_data))
            params_to_change = random.sample(list(self.params_data.keys()), k=number_of_mutations)

            for p in params_to_change:
                # Change mutation to pure random generation???
                mutation_strength = random.uniform(-2, 2)
                if self.params_data[p]["type"] != Y_N:
                    new_child.parameters[p] = self.params_data[p]["type"](new_child.parameters[p] *
                                                                          (1 + mutation_strength))
                    new_child.parameters[p] = max(new_child.parameters[p], self.params_data[p]["min"])
                    new_child.parameters[p] = min(new_child.parameters[p], self.params_data[p]["max"])

                if self.params_data[p]["type"] == float:
                    new_child.parameters[p] = round(new_child.parameters[p], self.params_data[p]["decimals"])

            new_child.parameters = self._params_constraints(new_child.parameters)

            if new_child.parameters not in self.population_params:
                offspring_pop.append(new_child)
                self.population_params.append(new_child.parameters)

        return offspring_pop

    def _params_constraints(self, params: typing.Dict) -> typing.Dict:

        if self.strategy == "obv":
            pass

        elif self.strategy == "sup_res":
            pass

        elif self.strategy == "ichimoku":
            params["kijun"] = max(params["kijun"], params["tenkan"])

        # elif self.strategy == "sma":
        #     params["slow_ma"] = max(params["slow_ma"], params["fast_ma"])

        elif self.strategy == "guppy":
            # Lengths cannot go over total number of candles in test
            num_candles = int(self.to_time - self.from_time + 60) // TF_SECONDS[self.tf]
            if num_candles <= 200:
                params["ema200_long"] = "n"
                params["ema200_short"] = "n"
            if params["mfi_period"] >= num_candles:
                params["mfi_period"] = random.randint(2, num_candles/2)
            if params["macd_fast"] >= num_candles:
                params["macd_fast"] = random.randint(2, num_candles/2)
            if params["macd_slow"] >= num_candles:
                params["macd_slow"] = random.randint(params["macd_fast"], num_candles)
            if params["rsi_length"] >= num_candles:
                params["rsi_length"] = random.randint(2, num_candles/2)
            if params["adx_smoothing"] >= num_candles:
                params["adx_smoothing"] = random.randint(2, num_candles/2)
            if params["adx_di_length"] >= num_candles:
                params["adx_di_length"] = random.randint(2, num_candles/2)
            if params["bb_length"] >= num_candles:
                params["bb_length"] = random.randint(2, num_candles/2)
            if params["wae_fast_length"] >= num_candles:
                params["wae_fast_length"] = random.randint(2, num_candles/2)
            if params["wae_slow_length"] >= num_candles:
                params["wae_slow_length"] = random.randint(params["wae_fast_length"], num_candles)
            if params["wae_bb_length"] >= num_candles:
                params["wae_bb_length"] = random.randint(2, num_candles/2)
            if params["wae_rma_length"] >= num_candles:
                params["wae_rma_length"] = random.randint(2, num_candles/2)

            # Don't turn off both Longs and Shorts
            if (params["trade_longs"].upper() == "N") and (params["trade_shorts"].upper() == "N"):
                choice = random.choice(["trade_longs", "trade_shorts"])
                params[choice] = "y"

            # WAE MACD Slow EMA must be larger than fast EMA
            if params["wae_slow_length"] < params["wae_fast_length"]:
                params["wae_slow_length"], params["wae_fast_length"] = params["wae_fast_length"], \
                                                                       params["wae_slow_length"]

            # TP1 < TP2 < TP3
            params["ltp1"] = min(params["ltp1"], params["ltp2"], params["ltp3"])
            params["ltp3"] = max(params["ltp1"], params["ltp2"], params["ltp3"])
            params["stp1"] = min(params["stp1"], params["stp2"], params["stp3"])
            params["stp3"] = max(params["stp1"], params["stp2"], params["stp3"])
            # Total quantity to take out cannot be more than 100%
            # if params["tp_long"] == 2 and (params["ltp1_qty"] == 100):
            #     while params["ltp1_qty"] == 100:
            #         params["ltp1_qty"] = round(random.uniform(params["ltp1_qty"]["min"], params["ltp1_qty"]["max"]),
            #                                    params["ltp1_qty"]["decimals"])
            # if params["tp_short"] == 2 and (params["stp1_qty"] == 100):
            #     while params["stp1_qty"] == 100:
            #         params["stp1_qty"] = round(random.uniform(params["stp1_qty"]["min"], params["stp1_qty"]["max"]),
            #                                    params["stp1_qty"]["decimals"])
            if params["tp_long"] == 3 and ((params["ltp1_qty"] + params["ltp2_qty"]) > 100):
                # while (params["ltp1_qty"] + params["ltp2_qty"]) >= 100:
                #     params["ltp1_qty"] = round(random.uniform(params["ltp1_qty"]["min"], params["ltp1_qty"]["max"]),
                #                                params["ltp1_qty"]["decimals"])
                #     params["ltp2_qty"] = round(random.uniform(params["ltp2_qty"]["min"], params["ltp2_qty"]["max"]),
                #                                params["ltp2_qty"]["decimals"])
                params["ltp2_qty"] = 100 - params["ltp1_qty"]
            if params["tp_short"] == 3 and ((params["stp1"] + params["stp2"]) > 100):
                # while (params["stp1_qty"] + params["stp2_qty"]) >= 100:
                #     params["stp1_qty"] = round(random.uniform(params["stp1_qty"]["min"], params["stp1_qty"]["max"]),
                #                                params["stp1_qty"]["decimals"])
                #     params["stp2_qty"] = round(random.uniform(params["stp2_qty"]["min"], params["stp2_qty"]["max"]),
                #                                params["stp2_qty"]["decimals"])
                params["stp2_qty"] = 100 - params["stp1_qty"]

            # Maintain safe R:R
            # if (params["sl_long"] == 0.1) and (params["ltp1"] < 0.4):
            #     params["ltp1"] = round(random.uniform(0.4, 5.0), 1)
            # elif (params["sl_long"] >= 0.2) and (params["sl_long"] <= 0.5) and \
            #         (params["ltp1"] < (3 * params["sl_long"])):
            #     params["ltp1"] = round(random.uniform(3 * params["sl_long"], 5.0), 1)
            # elif (params["sl_long"] >= 0.6) and (params["sl_long"] <= 2.5) and \
            #         (params["ltp1"] < (2 * params["sl_long"])):
            #     params["ltp1"] = round(random.uniform(2 * params["sl_long"], 5.0), 1)
            # elif params["sl_long"] > 3.1:
            #     params["sl_long"] = round(random.uniform(0.1, 2.5), 1)
            #     params["ltp1"] = round(random.uniform(2 * params["sl_long"], 5.0), 1)
            # if (params["sl_short"] == 0.1) and (params["stp1"] < 0.4):
            #     params["stp1"] = round(random.uniform(0.4, 5.0), 1)
            # elif (params["sl_short"] >= 0.2) and (params["sl_short"] <= 0.5) and \
            #         (params["stp1"] < (3 * params["sl_short"])):
            #     params["stp1"] = round(random.uniform(3 * params["sl_short"], 5.0), 1)
            # elif (params["sl_short"] >= 0.6) and (params["sl_short"] <= 2.5) and \
            #         (params["stp1"] < (2 * params["sl_short"])):
            #     params["stp1"] = round(random.uniform(2 * params["sl_short"], 5.0), 1)
            # elif params["sl_short"] > 2.5:
            #     params["sl_short"] = round(random.uniform(0.1, 2.5), 1)
            #     params["stp1"] = round(random.uniform(2 * params["sl_short"], 5.0), 1)

            # rr_long: float
            # rr_short: float
            # market_fee = 0.0006
            # limit_fee = 0.0001
            # i = 0
            # for i in range(20):
            #     bpl: int
            #     bps: int
            #     if 0 < params["sl_long"] < 0.2:
            #         bpl = 100
            #     elif 0.2 <= params["sl_long"] < 0.25:
            #         bpl = 80
            #     elif 0.25 <= params["sl_long"] < 0.33:
            #         bpl = 60
            #     elif 0.33 <= params["sl_long"] < 0.4:
            #         bpl = 50
            #     elif 0.4 <= params["sl_long"] < 0.5:
            #         bpl = 50
            #     elif 0.5 <= params["sl_long"] < 0.67:
            #         bpl = 30
            #     elif 0.67 <= params["sl_long"] < 0.8:
            #         bpl = 25
            #     elif 0.8 <= params["sl_long"] < 1.0:
            #         bpl = 20
            #     elif 1.0 <= params["sl_long"] < 1.25:
            #         bpl = 20
            #     elif 1.25 <= params["sl_long"] < 1.33:
            #         bpl = 15
            #     elif 1.33 <= params["sl_long"] < 1.67:
            #         bpl = 15
            #     elif 1.67 <= params["sl_long"] < 2.0:
            #         bpl = 10
            #     elif 2.0 <= params["sl_long"] <= 2.5:
            #         bpl = 10
            #     else:
            #         bpl = 1
            #     l_entry = bpl * market_fee
            #     l_win_fee = bpl * (1 + params["ltp1"] / 100) * limit_fee
            #     l_loss_fee = bpl * (1 - params["sl_long"] / 100) * market_fee
            #     long_reward = bpl * (params["ltp1"] / 100) - l_entry - l_win_fee
            #     long_risk = bpl * (params["sl_long"] / 100) + l_entry + l_loss_fee
            #     rr_long = round(long_reward / long_risk, 3)
            #
            #     if 0 < params["sl_short"] < 0.2:
            #         bps = 100
            #     elif 0.2 <= params["sl_short"] < 0.25:
            #         bps = 80
            #     elif 0.25 <= params["sl_short"] < 0.33:
            #         bps = 60
            #     elif 0.33 <= params["sl_short"] < 0.4:
            #         bps = 50
            #     elif 0.4 <= params["sl_short"] < 0.5:
            #         bps = 50
            #     elif 0.5 <= params["sl_short"] < 0.67:
            #         bps = 30
            #     elif 0.67 <= params["sl_short"] < 0.8:
            #         bps = 25
            #     elif 0.8 <= params["sl_short"] < 1.0:
            #         bps = 20
            #     elif 1.0 <= params["sl_short"] < 1.25:
            #         bps = 20
            #     elif 1.25 <= params["sl_short"] < 1.33:
            #         bps = 15
            #     elif 1.33 <= params["sl_short"] < 1.67:
            #         bps = 15
            #     elif 1.67 <= params["sl_short"] < 2.0:
            #         bps = 10
            #     elif 2.0 <= params["sl_short"] <= 2.5:
            #         bps = 10
            #     else:
            #         bps = 1
            #     s_entry = bps * market_fee
            #     s_win_fee = bps * (1 - params["stp1"] / 100) * limit_fee
            #     s_loss_fee = bps * (1 + params["sl_short"] / 100) * market_fee
            #     short_reward = bps * (params["stp1"] / 100) - s_entry - s_win_fee
            #     short_risk = bps * (params["sl_short"] / 100) + s_entry + s_loss_fee
            #     rr_min = 5.0
            #     rr_max = 10.0
            #     rr_short = round(short_reward / short_risk, 3)
            #     if (rr_min < rr_long) and (rr_long < rr_max) and (rr_min < rr_short) and (rr_short < rr_max):
            #         # print("RR ok")
            #         break
            #     if (rr_long > rr_max) or (rr_long < rr_min):
            #         params["sl_long"] = round(random.uniform(0.1, 2.5), 1)
            #         params["ltp1"] = round(random.uniform(0.2, 5.0), 1)
            #         # print(f"{rr_long} Recalculating RR long")
            #     if (rr_short > rr_max) or (rr_short < rr_min):
            #         params["sl_short"] = round(random.uniform(0.1, 2.5), 1)
            #         params["stp1"] = round(random.uniform(0.2, 5.0), 1)
            #         # print(f"{rr_short} Recalculating RR short")

            # # MFI multiplier must be large enough that YPOS doesn't skew signals to shorts
            # if params["mfi_mult"] < (params["mfi_ypos"] * 60):
            #     params["mfi_mult"] = params["mfi_ypos"] * 60

            # MACD Fast EMA must be shorter than Slow
            if params["macd_fast"] > params["macd_slow"]:
                params["macd_fast"], params["macd_slow"] = params["macd_slow"], params["macd_fast"]

        return params

    def crowding_distance(self, population: typing.List[BacktestResult]) -> typing.List[BacktestResult]:
        # v.4.1.5
        # for objective in ["pnl", "pnl_cv", "max_dd", "max_dd_cv"]:
        # for objective in ["pnl_dd_ratio", "pnl_dd_ratio_cv"]:
        # v.4.1.6
        for objective in ["pnl", "max_dd"]:
        # # v.4.1.7
        # for objective in ["pnl", "max_dd", "pnl_cv"]:
        # # v.4.1.8
        # for objective in ["pnl", "max_dd", "pnl_cv", "win_rate_cv"]:

            population = sorted(population, key=lambda x: getattr(x, objective))
            min_value = getattr(min(population, key=lambda x: getattr(x, objective)), objective)
            max_value = getattr(max(population, key=lambda x: getattr(x, objective)), objective)

            population[0].crowding_distance = float("inf")
            population[-1].crowding_distance = float("inf")

            for i in range(1, len(population) - 1):
                distance = getattr(population[i + 1], objective) - getattr(population[i - 1], objective)
                if ((max_value - min_value) != 0) and (max_value != float("inf")) and (max_value != -float("inf")) and \
                        (min_value != float("inf")) and (min_value != -float("inf")):
                    distance = distance / (max_value - min_value)
                population[i].crowding_distance += distance

        return population

    def non_dominated_sorting(self, population: typing.Dict[int, BacktestResult]) -> \
            typing.List[typing.List[BacktestResult]]:

        fronts = []

        for id_1, indiv_1 in population.items():
            for id_2, indiv_2 in population.items():

                # # Using pnl/dd ratio
                # if (indiv_1.pnl_dd_ratio >= indiv_2.pnl_dd_ratio) and \
                #     (indiv_1.pnl_dd_ratio_cv <= indiv_2.pnl_dd_ratio_cv) and \
                #     ((indiv_1.pnl_dd_ratio > indiv_2.pnl_dd_ratio) or
                #      (indiv_1.pnl_dd_ratio_cv < indiv_2.pnl_dd_ratio_cv)):
                #     indiv_1.dominates.append(id_2)
                # elif (indiv_2.pnl_dd_ratio >= indiv_1.pnl_dd_ratio) and \
                #         (indiv_2.pnl_dd_ratio_cv <= indiv_1.pnl_dd_ratio_cv) and \
                #         ((indiv_2.pnl_dd_ratio > indiv_1.pnl_dd_ratio) or
                #          (indiv_2.pnl_dd_ratio_cv < indiv_1.pnl_dd_ratio_cv)):
                #     indiv_1.dominated_by += 1

                # # v.4.1.5 pnl 1st
                # if indiv_1.pnl > indiv_2.pnl:
                #     if (indiv_1.max_dd <= indiv_2.max_dd) and (indiv_1.pnl_cv <= indiv_2.pnl_cv) and \
                #         (indiv_1.max_dd_cv <= indiv_2.max_dd_cv) and ((indiv_1.max_dd < indiv_2.max_dd) or
                #             (indiv_1.pnl_cv < indiv_2.pnl_cv) or (indiv_1.max_dd_cv < indiv_2.max_dd_cv)):
                #         indiv_1.dominates.append(id_2)
                # elif indiv_2.pnl > indiv_1.pnl:
                #     if (indiv_2.max_dd <= indiv_1.max_dd) and (indiv_2.pnl_cv <= indiv_1.pnl_cv) and \
                #         (indiv_2.max_dd_cv <= indiv_1.max_dd_cv) and ((indiv_2.max_dd < indiv_1.max_dd) or
                #             (indiv_2.pnl_cv < indiv_1.pnl_cv) or (indiv_2.max_dd_cv < indiv_1.max_dd_cv)):
                #         indiv_1.dominated_by += 1

                # # v.4.1.6
                # if indiv_1.pnl > indiv_2.pnl:
                #     if indiv_1.max_dd <= indiv_2.max_dd:
                #         indiv_1.dominates.append(id_2)
                # elif indiv_2.pnl > indiv_1.pnl:
                #     if indiv_2.max_dd <= indiv_1.max_dd:
                #         indiv_1.dominated_by += 1

                # # v.4.1.7
                # if indiv_1.pnl > indiv_2.pnl:
                #     if (indiv_1.max_dd <= indiv_2.max_dd) and (indiv_1.pnl_cv <= indiv_2.pnl_cv) and \
                #             ((indiv_1.max_dd < indiv_2.max_dd) or (indiv_1.pnl_cv < indiv_2.pnl_cv)):
                #         indiv_1.dominates.append(id_2)
                # elif indiv_2.pnl > indiv_1.pnl:
                #     if (indiv_2.max_dd <= indiv_1.max_dd) and (indiv_2.pnl_cv <= indiv_1.pnl_cv) and \
                #             ((indiv_2.max_dd < indiv_1.max_dd) or (indiv_2.pnl_cv < indiv_1.pnl_cv)):
                #         indiv_1.dominated_by += 1

                # # v.4.1.8
                # if indiv_1.pnl > indiv_2.pnl:
                #     if (indiv_1.max_dd <= indiv_2.max_dd) and (indiv_1.pnl_cv <= indiv_2.pnl_cv) and \
                #             ((indiv_1.max_dd < indiv_2.max_dd) or (indiv_1.pnl_cv < indiv_2.pnl_cv)):
                #         indiv_1.dominates.append(id_2)
                #     elif (indiv_1.max_dd <= indiv_2.max_dd) and (indiv_1.win_rate_cv <= indiv_2.win_rate_cv) and \
                #             ((indiv_1.max_dd < indiv_2.max_dd) or (indiv_1.win_rate_cv < indiv_2.win_rate_cv)):
                #         indiv_1.dominates.append(id_2)
                #     elif (indiv_1.pnl_cv <= indiv_2.pnl_cv) and (indiv_1.win_rate_cv <= indiv_2.win_rate_cv) and\
                #             ((indiv_1.pnl_cv < indiv_2.pnl_cv) or (indiv_1.win_rate_cv < indiv_2.win_rate_cv)):
                #         indiv_1.dominates.append(id_2)
                # elif indiv_2.pnl > indiv_1.pnl:
                #     if (indiv_2.max_dd <= indiv_1.max_dd) and (indiv_2.pnl_cv <= indiv_1.pnl_cv) and \
                #             ((indiv_2.max_dd < indiv_1.max_dd) or (indiv_2.pnl_cv < indiv_1.pnl_cv)):
                #         indiv_2.dominated_by += 1
                #     elif (indiv_2.max_dd <= indiv_1.max_dd) and (indiv_2.win_rate_cv <= indiv_1.win_rate_cv) and \
                #             ((indiv_2.max_dd < indiv_1.max_dd) or (indiv_2.win_rate_cv < indiv_1.win_rate_cv)):
                #         indiv_2.dominated_by += 1
                #     elif (indiv_2.pnl_cv <= indiv_1.pnl_cv) and (indiv_2.win_rate_cv <= indiv_1.win_rate_cv) and \
                #             ((indiv_2.pnl_cv < indiv_1.pnl_cv) or (indiv_2.win_rate_cv < indiv_1.win_rate_cv)):
                #         indiv_1.dominated_by += 1

                # # v4.1.9
                # if (indiv_1.pnl > indiv_2.pnl) and (((indiv_1.max_dd <= indiv_2.max_dd) and
                #                                      (indiv_1.pnl_cv <= indiv_2.pnl_cv) and
                #                                      ((indiv_1.max_dd < indiv_2.max_dd) or
                #                                       (indiv_1.pnl_cv < indiv_2.pnl_cv))) or
                #                                     ((indiv_1.max_dd <= indiv_2.max_dd) and
                #                                      (indiv_1.win_rate_cv <= indiv_2.win_rate_cv) and
                #                                      ((indiv_1.max_dd < indiv_2.max_dd) or
                #                                       (indiv_1.win_rate_cv < indiv_2.win_rate_cv))) or
                #                                     ((indiv_1.pnl_cv <= indiv_2.pnl_cv) and
                #                                      (indiv_1.win_rate_cv <= indiv_2.win_rate_cv) and
                #                                      ((indiv_1.pnl_cv < indiv_2.pnl_cv) or
                #                                       (indiv_1.win_rate_cv < indiv_2.win_rate_cv)))):
                #     indiv_1.dominates.append(id_2)
                # elif (indiv_2.pnl > indiv_1.pnl) and (((indiv_2.max_dd <= indiv_1.max_dd) and
                #                                        (indiv_2.pnl_cv <= indiv_1.pnl_cv) and
                #                                       ((indiv_2.max_dd < indiv_1.max_dd) or
                #                                        (indiv_2.pnl_cv < indiv_1.pnl_cv))) or
                #                                       ((indiv_2.max_dd <= indiv_1.max_dd) and
                #                                        (indiv_2.win_rate_cv <= indiv_1.win_rate_cv) and
                #                                       ((indiv_2.max_dd < indiv_1.max_dd) or
                #                                        (indiv_2.win_rate_cv < indiv_1.win_rate_cv))) or
                #                                       (indiv_2.pnl_cv <= indiv_1.pnl_cv) and
                #                                       (indiv_2.win_rate_cv <= indiv_1.win_rate_cv) and
                #                                       ((indiv_2.pnl_cv < indiv_1.pnl_cv) or
                #                                        (indiv_2.win_rate_cv < indiv_1.win_rate_cv))):
                #     indiv_1.dominated_by += 1

                # # OG / v.4.1.10
                # if indiv_1.pnl >= indiv_2.pnl and indiv_1.max_dd <= indiv_2.max_dd and \
                #     (indiv_1.pnl > indiv_2.pnl or indiv_1.max_dd < indiv_2.max_dd):
                #     indiv_1.dominates.append(id_2)
                # elif indiv_2.pnl >= indiv_1.pnl and indiv_2.max_dd <= indiv_1.max_dd and \
                #     (indiv_2.pnl > indiv_1.pnl or indiv_2.max_dd < indiv_1.max_dd):
                #     indiv_1.dominated_by += 1
                # if indiv_1.pnl >= indiv_2.pnl and indiv_1.max_dd <= indiv_2.max_dd and indiv_1.pnl_cv <= \
                #         indiv_2.pnl_cv \
                #     and (indiv_1.pnl_cv < indiv_2.pnl_cv or indiv_1.max_dd < indiv_2.max_dd):
                #     indiv_1.dominates.append(id_2)
                # elif indiv_2.pnl > indiv_1.pnl and indiv_2.max_dd <= indiv_1.max_dd and \
                #         indiv_1.pnl_cv <= indiv_2.pnl_cv and \
                #     (indiv_2.pnl_cv < indiv_1.pnl_cv or indiv_2.max_dd < indiv_1.max_dd):
                #     indiv_1.dominated_by += 1

                # v. 4.1.11
                if (indiv_1.pnl > indiv_2.pnl) and (((indiv_1.max_dd <= indiv_2.max_dd) and
                                                     (indiv_1.win_rate >= indiv_2.win_rate) and
                                                     ((indiv_1.max_dd < indiv_2.max_dd) or
                                                      (indiv_1.win_rate > indiv_2.win_rate))) or
                                                    ((indiv_1.max_dd <= indiv_2.max_dd) and
                                                     (indiv_1.num_trades >= indiv_2.num_trades) and
                                                     ((indiv_1.max_dd < indiv_2.max_dd) or
                                                      (indiv_1.num_trades > indiv_2.num_trades))) or
                                                    ((indiv_1.win_rate >= indiv_2.win_rate) and
                                                     (indiv_1.num_trades >= indiv_2.num_trades) and
                                                     ((indiv_1.win_rate > indiv_2.win_rate) or
                                                      (indiv_1.num_trades > indiv_2.num_trades)))):
                    indiv_1.dominates.append(id_2)
                elif (indiv_2.pnl > indiv_1.pnl) and (((indiv_2.max_dd <= indiv_1.max_dd) and
                                                       (indiv_2.win_rate >= indiv_1.win_rate) and
                                                      ((indiv_2.max_dd < indiv_1.max_dd) or
                                                       (indiv_2.win_rate > indiv_1.win_rate))) or
                                                      ((indiv_2.max_dd <= indiv_1.max_dd) and
                                                       (indiv_2.num_trades >= indiv_1.num_trades) and
                                                      ((indiv_2.max_dd < indiv_1.max_dd) or
                                                       (indiv_2.num_trades > indiv_1.num_trades))) or
                                                      (indiv_2.win_rate >= indiv_1.win_rate) and
                                                      (indiv_2.num_trades >= indiv_1.num_trades) and
                                                      ((indiv_2.win_rate > indiv_1.win_rate) or
                                                       (indiv_2.num_trades > indiv_1.num_trades))):
                    indiv_1.dominated_by += 1

                # # v 4.1.11a
                # if (indiv_1.pnl > indiv_2.pnl) and ((indiv_1.max_dd <= indiv_2.max_dd) and
                #                                     (indiv_1.num_trades >= indiv_2.num_trades) and
                #                                     (indiv_1.win_rate >= indiv_2.win_rate)) and \
                #                                    ((indiv_1.max_dd < indiv_2.max_dd) or
                #                                     (indiv_1.num_trades > indiv_2.num_trades) or
                #                                     (indiv_1.win_rate > indiv_2.win_rate)):
                #     indiv_1.dominates.append(id_2)
                # elif (indiv_2.pnl > indiv_1.pnl) and ((indiv_2.max_dd <= indiv_1.max_dd) and
                #                                       (indiv_2.num_trades >= indiv_1.num_trades) and
                #                                       (indiv_2.win_rate >= indiv_1.win_rate)) and \
                #                                      ((indiv_2.max_dd < indiv_1.max_dd) or
                #                                       (indiv_2.num_trades > indiv_1.num_trades) or
                #                                       (indiv_2.win_rate > indiv_1.win_rate)):
                #     indiv_1.dominated_by += 1

                # # v.4.1.12
                # if indiv_1.pnl > indiv_2.pnl:
                #     indiv_1.dominates.append(id_2)
                # elif indiv_2.pnl > indiv_1.pnl:
                #     indiv_1.dominated_by += 1

                # # v. 4.1.13
                # if (indiv_1.pnl > indiv_2.pnl) and ((indiv_1.max_dd <= indiv_2.max_dd) and
                #                                     (indiv_1.num_trades >= indiv_2.num_trades)) and \
                #                                     ((indiv_1.max_dd < indiv_2.max_dd) or
                #                                     (indiv_1.num_trades > indiv_2.num_trades)):
                #     indiv_1.dominates.append(id_2)
                # elif (indiv_2.pnl > indiv_1.pnl) and ((indiv_2.max_dd <= indiv_1.max_dd) and
                #                                       (indiv_2.num_trades >= indiv_1.num_trades)) and \
                #                                       ((indiv_2.max_dd < indiv_1.max_dd) or
                #                                       (indiv_2.num_trades > indiv_1.num_trades)):
                #     indiv_1.dominated_by += 1

            if indiv_1.dominated_by == 0:
                if len(fronts) == 0:
                    fronts.append([])
                fronts[0].append(indiv_1)
                indiv_1.rank = 0

        i = 0

        while True:
            fronts.append([])

            for indiv_1 in fronts[i]:
                for indiv_2_id in indiv_1.dominates:
                    population[indiv_2_id].dominated_by -= 1
                    if population[indiv_2_id].dominated_by == 0:
                        fronts[i + 1].append(population[indiv_2_id])
                        population[indiv_2_id].rank = i + 1

            if len(fronts[i + 1]) > 0:
                i += 1
            else:
                del fronts[-1]
                break

        return fronts

    def evaluate_population(self, population: typing.List[BacktestResult]) -> typing.List[BacktestResult]:
        if self.strategy == "obv":

            for bt in population:
                bt.pnl, bt.max_dd = strategies.obv.backtest(self.data, ma_period=bt.parameters["ma_period"])
                if bt.pnl == 0:
                    bt.pnl = -float("inf")
                    bt.max_dd = float("inf")
            return population

        elif self.strategy == "ichimoku":

            for bt in population:
                bt.pnl, bt.max_dd = strategies.ichimoku.backtest(self.data, tenkan_period=bt.parameters["tenkan"],
                                                                 kijun_period=bt.parameters["kijun"])

                if bt.pnl == 0:
                    bt.pnl = -float("inf")
                    bt.max_dd = float("inf")
            return population

        elif self.strategy == "sup_res":

            for bt in population:
                bt.pnl, bt.max_dd = \
                    strategies.support_resistance.backtest(self.data, min_points=bt.parameters["min_points"],
                                                           min_diff_points=bt.parameters["min_diff_points"],
                                                           rounding_nb=bt.parameters["rounding_nb"],
                                                           take_profit=bt.parameters["take_profit"],
                                                           stop_loss=bt.parameters["stop_loss"])

                if bt.pnl == 0:
                    bt.pnl = -float("inf")
                    bt.max_dd = float("inf")

            return population

        # elif self.strategy == "mfi":
        #     bt.pnl, bt.max_dd = strategies.mfi.backtest(self.data, period=params['period'],
        #                                                 multiplier=params['multiplier'], ypos=params['ypos'])

        # elif self.strategy == "sma":
        #     self.lib.Sma_execute_backtest(self.obj, bt.parameters["slow_ma"], bt.parameters["fast_ma"])
        #     bt.pnl = self.lib.Sma_get_pnl(self.obj)
        #     bt.max_dd = self.lib.Sma_get_max_dd(self.obj)
        #
        #     if bt.pnl == 0:
        #         bt.pnl = -float("inf")
        #         bt.max_dd = float("inf")
        #
        #     return population

        elif self.strategy == "guppy":
            for bt in population:
                bt.pnl, bt.max_dd, bt.win_rate, bt.rr_long, bt.rr_short, bt.num_trades, bt.mod_win_rate, \
                    bt.max_losses, bt.max_wins, bt.num_longs, bt.num_shorts \
                    = strategies.guppy.backtest(df=self.data, initial_capital=self.initial_capital,
                                                trade_longs=bt.parameters['trade_longs'],
                                                trade_shorts=bt.parameters['trade_shorts'],
                                                sl_long=bt.parameters['sl_long'],
                                                sl_short=bt.parameters['sl_short'], mfi_long=bt.parameters['mfi_long'],
                                                mfi_short=bt.parameters['mfi_short'],
                                                mfi_period=bt.parameters['mfi_period'],
                                                mfi_mult=bt.parameters['mfi_mult'], mfi_ypos=bt.parameters['mfi_ypos'],
                                                mfi_long_threshold=bt.parameters['mfi_long_threshold'],
                                                mfi_short_threshold=bt.parameters['mfi_short_threshold'],
                                                macd_long=bt.parameters['macd_long'],
                                                macd_short=bt.parameters['macd_short'],
                                                macd_fast=bt.parameters['macd_fast'],
                                                macd_slow=bt.parameters['macd_slow'],
                                                macd_signal=bt.parameters['macd_signal'],
                                                rsi_long=bt.parameters['rsi_long'],
                                                rsi_short=bt.parameters['rsi_short'],
                                                rsi_length=bt.parameters['rsi_length'],
                                                rsi_overbought=bt.parameters['rsi_overbought'],
                                                rsi_oversold=bt.parameters['rsi_oversold'],
                                                ema200_long=bt.parameters['ema200_long'],
                                                ema200_short=bt.parameters['ema200_short'],
                                                guppy_fast_long=bt.parameters['guppy_fast_long'],
                                                guppy_fast_short=bt.parameters['guppy_fast_short'],
                                                ribbon_check_long=bt.parameters['ribbon_check_long'],
                                                ribbon_check_short=bt.parameters['ribbon_check_short'],
                                                sl_type_long=bt.parameters['sl_type_long'],
                                                sl_type_short=bt.parameters['sl_type_short'],
                                                min_rr_long=bt.parameters['min_rr_long'],
                                                min_rr_short=bt.parameters['min_rr_short'],
                                                tsl_size_long=bt.parameters['tsl_size_long'],
                                                tsl_size_short=bt.parameters['tsl_size_short'],
                                                band6_cushion_long=bt.parameters['band6_cushion_long'],
                                                band6_cushion_short=bt.parameters['band6_cushion_short'],
                                                gsl_moveto_long=bt.parameters['gsl_moveto_long'],
                                                gsl_moveto_short=bt.parameters['gsl_moveto_short'],
                                                move_sl_type_long=bt.parameters['move_sl_type_long'],
                                                move_sl_type_short=bt.parameters['move_sl_type_short'],
                                                move_sl_long=bt.parameters['move_sl_long'],
                                                move_sl_short=bt.parameters['move_sl_short'],
                                                risk=bt.parameters['risk'],
                                                leverage=bt.parameters['leverage'], tp_long=bt.parameters['tp_long'],
                                                tp_short=bt.parameters['tp_short'], ltp1=bt.parameters['ltp1'],
                                                ltp1_qty=bt.parameters['ltp1_qty'], ltp2=bt.parameters['ltp2'],
                                                ltp2_qty=bt.parameters['ltp2_qty'], ltp3=bt.parameters['ltp3'],
                                                stp1=bt.parameters['stp1'], stp1_qty=bt.parameters['stp1_qty'],
                                                stp2=bt.parameters['stp2'], stp2_qty=bt.parameters['stp2_qty'],
                                                stp3=bt.parameters['stp3'], mode="o", contract=self.contract,
                                                tf=self.tf, from_time=self.from_time, to_time=self.to_time,
                                                bb_long=bt.parameters['bb_long'], bb_short=bt.parameters['bb_short'],
                                                bb_length=bt.parameters['bb_length'], bb_mult=bt.parameters['bb_mult'],
                                                wae_long=bt.parameters['wae_long'],
                                                wae_short=bt.parameters['wae_short'],
                                                wae_sensitivity=bt.parameters['wae_sensitivity'],
                                                wae_fast_length=bt.parameters['wae_fast_length'],
                                                wae_slow_length=bt.parameters['wae_slow_length'],
                                                wae_bb_length=bt.parameters['wae_bb_length'],
                                                wae_bb_mult=bt.parameters['wae_bb_mult'],
                                                wae_rma_length=bt.parameters['wae_rma_length'],
                                                wae_dz_mult=bt.parameters['wae_dz_mult'],
                                                wae_expl_check=bt.parameters['wae_expl_check'],
                                                adx_long=bt.parameters['adx_long'],
                                                adx_short=bt.parameters['adx_short'],
                                                adx_smoothing=bt.parameters['adx_smoothing'],
                                                adx_di_length=bt.parameters['adx_di_length'],
                                                adx_length_long=bt.parameters['adx_length_long'],
                                                adx_length_short=bt.parameters['adx_length_short'],)

                # if bt.pnl == 0:
                #     bt.pnl = -float("inf")
                #     bt.max_dd = float("inf")

                bt.pnl_history.append(bt.pnl)
                bt.max_dd_history.append(bt.max_dd)
                bt.win_rate_history.append(bt.win_rate)
                if bt.max_dd != 0:
                    bt.pnl_dd_ratio = bt.pnl / bt.max_dd
                else:
                    bt.pnl_dd_ratio = 0
                bt.pnl_dd_ratio_history.append(bt.pnl_dd_ratio)
                # print(f"PNL: {bt.pnl_history}\n MaxDD: {bt.max_dd_history}\n WR: {bt.win_rate_history}")

            return population

    def rss_period(self) -> pd.DataFrame:
        h5_db = Hdf5Client()
        data_start, data_end = h5_db.get_first_last_timestamp(self.contract)
        time_delta = int(self.to_time - self.from_time + 60)
        period_start = random.randint((data_start + (60 * TF_SECONDS[self.tf])), (data_end - time_delta))
        period_start -= (60 * TF_SECONDS[self.tf])
        data = h5_db.get_data(self.contract, period_start, (period_start + time_delta))
        data = resample_timeframe(data, self.tf)
        logger.debug(f"RSS period: {data.index[0]} to {data.index[-1]}")

        return data

    def rss_backtest(self, data: pd.DataFrame, population: typing.Dict[int, BacktestResult]) -> \
            typing.Dict[int, BacktestResult]:
        rss_pop = copy.deepcopy(population)

        i = 0
        for i in range(len(rss_pop)):

            rss_pop[i].pnl, rss_pop[i].max_dd, rss_pop[i].win_rate, rss_pop[i].rr_long, rss_pop[i].rr_short, \
                rss_pop[i].num_trades, rss_pop[i].mod_win_rate, rss_pop[i].max_losses, rss_pop[i].max_wins, \
                rss_pop[i].num_longs, rss_pop[i].num_shorts \
                = strategies.guppy.backtest(df=data, initial_capital=self.initial_capital,
                                            trade_longs=rss_pop[i].parameters['trade_longs'],
                                            trade_shorts=rss_pop[i].parameters['trade_shorts'],
                                            sl_long=rss_pop[i].parameters['sl_long'],
                                            sl_short=rss_pop[i].parameters['sl_short'],
                                            mfi_long=rss_pop[i].parameters['mfi_long'],
                                            mfi_short=rss_pop[i].parameters['mfi_short'],
                                            mfi_period=rss_pop[i].parameters['mfi_period'],
                                            mfi_mult=rss_pop[i].parameters['mfi_mult'],
                                            mfi_ypos=rss_pop[i].parameters['mfi_ypos'],
                                            mfi_long_threshold=rss_pop[i].parameters['mfi_long_threshold'],
                                            mfi_short_threshold=rss_pop[i].parameters['mfi_short_threshold'],
                                            macd_long=rss_pop[i].parameters['macd_long'],
                                            macd_short=rss_pop[i].parameters['macd_short'],
                                            macd_fast=rss_pop[i].parameters['macd_fast'],
                                            macd_slow=rss_pop[i].parameters['macd_slow'],
                                            macd_signal=rss_pop[i].parameters['macd_signal'],
                                            rsi_long=rss_pop[i].parameters['rsi_long'],
                                            rsi_short=rss_pop[i].parameters['rsi_short'],
                                            rsi_length=rss_pop[i].parameters['rsi_length'],
                                            rsi_overbought=rss_pop[i].parameters['rsi_overbought'],
                                            rsi_oversold=rss_pop[i].parameters['rsi_oversold'],
                                            ema200_long=rss_pop[i].parameters['ema200_long'],
                                            ema200_short=rss_pop[i].parameters['ema200_short'],
                                            guppy_fast_long=rss_pop[i].parameters['guppy_fast_long'],
                                            guppy_fast_short=rss_pop[i].parameters['guppy_fast_short'],
                                            ribbon_check_long=rss_pop[i].parameters['ribbon_check_long'],
                                            ribbon_check_short=rss_pop[i].parameters['ribbon_check_short'],
                                            sl_type_long=rss_pop[i].parameters['sl_type_long'],
                                            sl_type_short=rss_pop[i].parameters['sl_type_short'],
                                            min_rr_long=rss_pop[i].parameters['min_rr_long'],
                                            min_rr_short=rss_pop[i].parameters['min_rr_short'],
                                            tsl_size_long=rss_pop[i].parameters['tsl_size_long'],
                                            tsl_size_short=rss_pop[i].parameters['tsl_size_short'],
                                            band6_cushion_long=rss_pop[i].parameters['band6_cushion_long'],
                                            band6_cushion_short=rss_pop[i].parameters['band6_cushion_short'],
                                            gsl_moveto_long=rss_pop[i].parameters['gsl_moveto_long'],
                                            gsl_moveto_short=rss_pop[i].parameters['gsl_moveto_short'],
                                            move_sl_type_long=rss_pop[i].parameters['move_sl_type_long'],
                                            move_sl_type_short=rss_pop[i].parameters['move_sl_type_short'],
                                            move_sl_long=rss_pop[i].parameters['move_sl_long'],
                                            move_sl_short=rss_pop[i].parameters['move_sl_short'],
                                            risk=rss_pop[i].parameters['risk'],
                                            leverage=rss_pop[i].parameters['leverage'],
                                            tp_long=rss_pop[i].parameters['tp_long'],
                                            tp_short=rss_pop[i].parameters['tp_short'],
                                            ltp1=rss_pop[i].parameters['ltp1'],
                                            ltp1_qty=rss_pop[i].parameters['ltp1_qty'],
                                            ltp2=rss_pop[i].parameters['ltp2'],
                                            ltp2_qty=rss_pop[i].parameters['ltp2_qty'],
                                            ltp3=rss_pop[i].parameters['ltp3'],
                                            stp1=rss_pop[i].parameters['stp1'],
                                            stp1_qty=rss_pop[i].parameters['stp1_qty'],
                                            stp2=rss_pop[i].parameters['stp2'],
                                            stp2_qty=rss_pop[i].parameters['stp2_qty'],
                                            stp3=rss_pop[i].parameters['stp3'], mode="o", contract=self.contract,
                                            tf=self.tf, from_time=self.from_time, to_time=self.to_time,
                                            bb_long=rss_pop[i].parameters['bb_long'],
                                            bb_short=rss_pop[i].parameters['bb_short'],
                                            bb_length=rss_pop[i].parameters['bb_length'],
                                            bb_mult=rss_pop[i].parameters['bb_mult'],
                                            wae_long=rss_pop[i].parameters['wae_long'],
                                            wae_short=rss_pop[i].parameters['wae_short'],
                                            wae_sensitivity=rss_pop[i].parameters['wae_sensitivity'],
                                            wae_fast_length=rss_pop[i].parameters['wae_fast_length'],
                                            wae_slow_length=rss_pop[i].parameters['wae_slow_length'],
                                            wae_bb_length=rss_pop[i].parameters['wae_bb_length'],
                                            wae_bb_mult=rss_pop[i].parameters['wae_bb_mult'],
                                            wae_rma_length=rss_pop[i].parameters['wae_rma_length'],
                                            wae_dz_mult=rss_pop[i].parameters['wae_dz_mult'],
                                            wae_expl_check=rss_pop[i].parameters['wae_expl_check'],
                                            adx_long=rss_pop[i].parameters['adx_long'],
                                            adx_short=rss_pop[i].parameters['adx_short'],
                                            adx_smoothing=rss_pop[i].parameters['adx_smoothing'],
                                            adx_di_length=rss_pop[i].parameters['adx_di_length'],
                                            adx_length_long=rss_pop[i].parameters['adx_length_long'],
                                            adx_length_short=rss_pop[i].parameters['adx_length_short'],)

            # if rss_pop[i].pnl == 0:
            #     rss_pop[i].pnl = -float("inf")
            #     rss_pop[i].max_dd = float("inf")

            population[i].pnl_history.append(rss_pop[i].pnl)
            population[i].max_dd_history.append(rss_pop[i].max_dd)
            population[i].win_rate_history.append(rss_pop[i].win_rate)
            if rss_pop[i].max_dd != 0:
                rss_pop[i].pnl_dd_ratio = rss_pop[i].pnl / rss_pop[i].max_dd
            else:
                rss_pop[i].pnl_dd_ratio = 0
            population[i].pnl_dd_ratio_history.append(rss_pop[i].pnl_dd_ratio)

            # print(f"PNL: {population[i].pnl_history}\nMaxDD: {population[i].max_dd_history}\nWR: "
            #       f"{population[i].win_rate_history}")

        return rss_pop

    def cv_calculation(self, population: typing.List[BacktestResult], rss_pop: typing.List[BacktestResult]) -> \
            typing.List[BacktestResult]:

        for i in range(len(population)):
            avg_pnl_dd_ratio = np.mean(population[i].pnl_dd_ratio_history)
            std_pnl_dd_ratio = np.std(population[i].pnl_dd_ratio_history)
            if avg_pnl_dd_ratio != 0:
                if avg_pnl_dd_ratio < 0:
                    # Decrease priority for cases where results were poor in both populations
                    population[i].pnl_dd_ratio_cv = std_pnl_dd_ratio / avg_pnl_dd_ratio * -1
                else:
                    if avg_pnl_dd_ratio > population[i].pnl_dd_ratio:
                        # Increase priority for cases where results were better in RSS test average
                        population[i].pnl_dd_ratio_cv = std_pnl_dd_ratio / avg_pnl_dd_ratio * -1
                    else:
                        population[i].pnl_dd_ratio_cv = std_pnl_dd_ratio / avg_pnl_dd_ratio
            else:
                population[i].pnl_dd_ratio_cv = float("inf")

            avg_pnl = np.mean(population[i].pnl_history)
            std_pnl = np.std(population[i].pnl_history)
            if avg_pnl != 0:
                if avg_pnl < 0:
                    # Decrease priority for cases where results were poor in both populations
                    population[i].pnl_cv = std_pnl / avg_pnl * -1
                else:
                    if avg_pnl > population[i].pnl:
                        # Increase priority for cases where results were better in RSS population
                        population[i].pnl_cv = std_pnl / avg_pnl * -1
                    else:
                        population[i].pnl_cv = std_pnl / avg_pnl
            else:
                # Bad if no PNL - never traded
                population[i].pnl_cv = float("inf")

            avg_max_dd = np.mean(population[i].max_dd_history)
            std_max_dd = np.std(population[i].max_dd_history)
            if avg_max_dd != 0:
                # Increase priority for cases where results were better in RSS population
                if avg_max_dd < population[i].pnl:
                    population[i].max_dd_cv = std_max_dd / avg_max_dd * -1
                else:
                    population[i].max_dd_cv = std_max_dd / avg_max_dd
            else:
                # If PNL also 0, did not trade
                if population[i].pnl_cv == float("inf"):
                    population[i].max_dd_cv = float("inf")
                else:
                    # If no Max DD (not necessarily a bad thing), CV = 0
                    population[i].max_dd_cv = 0

            avg_win_rate = np.mean(population[i].win_rate_history)
            std_win_rate = np.std(population[i].win_rate_history)
            if avg_win_rate != 0:
                # Increase priority for cases where results were better in RSS population
                if avg_win_rate > population[i].win_rate:
                    population[i].win_rate_cv = std_win_rate / avg_win_rate * -1
                else:
                    population[i].win_rate_cv = std_win_rate / avg_win_rate
            else:
                # Bad if 0 Win Rate
                population[i].win_rate_cv = float("inf")

        return population
