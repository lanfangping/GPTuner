from abc import abstractmethod
from config_recommender.workload_runner import BenchbaseRunner
import time
import sys
import os
import json
import threading
import glob
import re
import time
import random
import functools
from ConfigSpace import (
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from search_space.knowledge_based_space import unify_unit


class DefaultSpace:
    """ Base template of GPTuner"""
    def __init__(self, dbms, test, timeout, target_knobs_path, results_folder, seed, error_log):
        self.dbms = dbms
        self.seed = seed if seed is not None else 1
        self.test = test
        self.timeout = timeout
        self.target_knobs_path = target_knobs_path
        self.round = 0
        task_folder = os.path.dirname(os.path.dirname(results_folder))
        self.summary_path = os.path.join(task_folder, 'temp_results') # "./optimization_results/temp_results"
        self.benchmark_copy_db = ['tpcc', 'twitter', "sibench", "voter", "tatp", "smallbank", "seats"]   # Some benchmark will insert or delete data, Need to be rewrite each time.
        self.benchmark_latency = ['tpch']
        self.factors = [0, 0.25, 0.5]
        self.search_space = ConfigurationSpace()
        self.skill_path = os.path.join(task_folder, f"knowledge_collection/{self.dbms.name}/structured_knowledge/normal")
        self.target_knobs = self.knob_select()
        if self.test in self.benchmark_copy_db:
            self.dbms.create_template(self.test)
        # self.penalty = 0
        self.penalty = self.get_default_result()
        print(f"DEFAULT : {self.penalty}")
        self.log_file = os.path.join(task_folder, f"{self.dbms.name}/log/{self.seed}_log.txt")
        self.feasible_configs = {}
        self.feasible_configs_path = os.path.join(results_folder, f"{self.seed}/feasible_configs.json")
        self.init_log_file()
        self.prev_end = 0
        self.error_log = error_log


    def init_log_file(self):
        with open(self.log_file, 'w') as file:
            file.write(f"Round\tStart\tEnd\tBenchmark_Elapsed\tTuning_overhead\n")

    def _log(self, begin_time, end_time):
        if self.round == 1:
            self.prev_end = begin_time
        with open(self.log_file, 'a') as file:
            file.write(f"{self.round}\t{begin_time}\t{end_time}\t{end_time-begin_time}\t{begin_time-self.prev_end}\n")
        self.prev_end = end_time

    def _transfer_unit(self, value):
        value = str(value)
        value = value.replace(" ", "")
        value = value.replace(",", "")
        if value.isalpha():
            value = "1" + value
        # pattern = r'(\d+\.\d+|\d+)([a-zA-Z]+)'
        pattern = r'(\d+(?:\.\d+)?)[\s]*([a-zA-Z]+)'
        match = re.match(pattern, value)
        if not match:
            return float(value)
        number, unit = match.group(1), match.group(2)
        unit_to_size = {
            'kB': 1e3,
            'KB': 1e3,
            'MB': 1e6,
            'GB': 1e9,
            'TB': 1e12,
            'K': 1e3,
            'M': 1e6,
            'G': 1e9,
            'B': 1,
            'ms': 1,
            's': 1000,
            'min': 60000,
            'h': 60 * 60000,
            'hour': 60 * 60000,
            'day': 24 * 60 * 60000,
            'million': 1e6
        }
        if unit in unit_to_size.keys():
            return float(number) * unit_to_size[unit]
        else:
            print(f"unknown unit '{unit}', return 1")
            return 1
    
    def _type_transfer(self, knob_type, value):
        value = str(value)
        value = value.replace(",", "")
        if knob_type == "integer":
            return int(round(float(value)))
        if knob_type == "real":
            return float(value)

    def _set_feasible(self, feasible=True, run_crash=False):
        "set whether the proposed config is valid"

        self.feasible_configs[self.round-1] = {'feasible': feasible, 'run_crash': run_crash, 'penalty': self.penalty}
        with open(self.feasible_configs_path, 'w') as f:
            json.dump(self.feasible_configs, f)

    def knob_select(self):
        """ 
            Select which knobs to be tuned, store the names in 'self.target_knobs' 
            Default implementation is to use fixed knobs. Provide the path to the file containing the knobs' names.
        """
        current_directory = os.getcwd()
        print(current_directory)
        with open(self.target_knobs_path, 'r') as file:
            lines = file.readlines()
        candidate_knobs = [line.strip() for line in lines]
        target_knobs = []
        for knob in candidate_knobs:
            if knob not in self.dbms.knob_info.keys() or "vartype" not in self.dbms.knob_info[knob] or self.dbms.knob_info[knob]["vartype"] == "string":
                continue
            else:
                target_knobs.append(knob)
        return target_knobs
    
    def get_default_space(self, knob_name, info):
        boot_value = info["reset_val"]
        min_value = info["min_val"]
        max_value = info["max_val"]
        knob_type = info["vartype"]
        if knob_type == "integer":
            if int(max_value) > sys.maxsize:
                knob = UniformIntegerHyperparameter(
                    knob_name, 
                    int(int(min_value) / 1000), 
                    int(int(max_value) / 1000),
                    default_value = int(int(boot_value) / 1000)
                )
            else:
                if min_value == max_value: # the parameter has the equal min_value and max_value
                    knob = Constant(knob, int(max_value))
                else:
                    knob = UniformIntegerHyperparameter(
                        knob_name,
                        int(min_value),
                        int(max_value),
                        default_value = int(boot_value),
                    )
        elif knob_type == "real":
            knob = UniformFloatHyperparameter(
                knob_name,
                float(min_value),
                float(max_value),
                default_value = float(boot_value)
            )
        elif knob_type == "enum":
            knob = CategoricalHyperparameter(
                knob_name,
                [str(enum_val) for enum_val in info["enumvals"]],
                default_value = str(boot_value),
            )
        elif knob_type == "bool":
            knob = CategoricalHyperparameter(
                knob_name,
                ["on", "off"],
                default_value = str(boot_value)
            )
        return knob

    def get_default_result(self):
        print("Test the result in default conf")
        dbms = self.dbms
        print(f"--- Restore the dbms to default configuration ---")
        dbms.reset_config()
        dbms.reconfigure()

        try:
            if self.test in self.benchmark_copy_db:
            # reload the data
                print("Reloading the data")
                dbms._disconnect()
                dbms._connect(f"{self.test}_template")
                dbms.copy_db(target_db="benchbase", source_db=f"{self.test}_template")
                print("Reloading completed")
                time.sleep(12)
                dbms._disconnect()
                time.sleep(4)
                dbms._connect('benchbase')
                time.sleep(3)
                pass
                
            print("Begin to run benchbase...")
            runner = BenchbaseRunner(dbms=dbms, test=self.test, target_path=self.summary_path)
            runner.clear_summary_dir()
            t = threading.Thread(target=runner.run_benchmark)
            t.start()
            t.join()
            throughput, average_latency = runner.get_throughput(), runner.get_latency()
        except Exception as e:
            print(f'Exception for {self.test}: {e}')

        if self.test not in self.benchmark_latency:
            return throughput
        else:
            return average_latency


    def set_and_replay(self, config, seed=0):
        # return random.uniform(1000,2000)
        begin_time = time.time()
        cost = self.set_and_replay_ori(config, seed)
        end_time = time.time()
        self._log(begin_time, end_time)
        return cost


    def set_and_replay_ori(self, config, seed=0):
        self.round += 1 # start from 1
        print(f"Tuning round {self.round} ...")
        dbms = self.dbms
        print(f"--- Restore the dbms to default configuration ---")
        dbms.reset_config()
        dbms.reconfigure()
        # reload the data
        if self.test in self.benchmark_copy_db:
            print("Reloading the data")
            dbms._disconnect()
            dbms._connect(f"{self.test}_template")
            dbms.copy_db(target_db="benchbase", source_db=f"{self.test}_template")
            print("Reloading completed")
            time.sleep(12)
            dbms._disconnect()
            time.sleep(4)
            dbms._connect('benchbase')
            time.sleep(3)

        print(f"--- knob setting procedure ---")
        for knob in self.target_knobs:
            try:
                control_para = config[f"control_{knob}"]
                if control_para == "0":
                    value = config[knob]       
                elif control_para == "1":
                    value = config[f"special_{knob}"]
            except:
                value = config[knob]
            if not dbms.set_knob(knob, value):
                self.error_log.error(f"Config {self.round-1}: knob {knob} is failed to set to {value}")
            
        dbms.reconfigure()
        if self.test not in self.benchmark_latency:
            if dbms.failed_times == 4:
                self._set_feasible(feasible=False, run_crash=None)
                return -int(self.penalty) / 2
        else:
            if dbms.failed_times == 4:
                self._set_feasible(feasible=False, run_crash=None)
                return self.penalty * 2
            
        try:
            print("Begin to run benchbase...")
            runner = BenchbaseRunner(dbms=dbms, test=self.test, target_path=self.summary_path)
            runner.clear_summary_dir()
            t = threading.Thread(target=runner.run_benchmark)
            t.start()
            t.join(timeout=self.timeout)
            if t.is_alive():
                print("Benchmark is still running. Terminate it now.")
                runner.process.terminate()
                time.sleep(2)
                raise RuntimeError("Benchmark is still running. Terminate it now.") 
            else:
                print("Benchmark has finished.")
                if runner.check_sequence_in_file():  ### 如果query出错
                    raise RuntimeError("ERROR in Query.") 
                throughput, average_latency = runner.get_throughput(), runner.get_latency()

                if self.test not in self.benchmark_latency and throughput < self.penalty:
                    self.penalty = throughput
                if self.test in self.benchmark_latency and average_latency > self.penalty:
                    self.penalty = average_latency

        except Exception as e:
            print(f'Exception for {self.test}: {e}')
            self._set_feasible(feasible=False, run_crash=True)
            # update worst_perf
            if self.test not in self.benchmark_latency:
                return -int(self.penalty) / 2
                ###tpch
            else:
                return self.penalty * 2
    
        self._set_feasible(feasible=True, run_crash=False)
        if self.test not in self.benchmark_latency:
            return -throughput
        else:
            return average_latency


    @abstractmethod
    def define_search_space(self):
        pass


    def _get_value_and_unit(self, value_with_unit):
        if value_with_unit is None:
            return None, None
        pattern = r'(\d+(?:\.\d+)?)[\s]*([a-zA-Z]+)' # r'(\d+(?:\.\d+)?)[\s]*([a-zA-Z]+)'. 
        match = re.match(pattern, str(value_with_unit))
        if match:
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else None
            return value, unit
        else:
            print(f"Value '{value_with_unit}' does not match the expected pattern. Returning the original value.")
            return value_with_unit, None

    def get_special_info(self):
        self.suggest_knob_info = {}
        normal_skill_path = os.path.join(os.path.dirname(self.special_skill_path), 'normal')
        for file_name in os.listdir(normal_skill_path):
            with open(os.path.join(normal_skill_path, file_name), 'r') as json_file:
                normal_skill = json.load(json_file)

                # Assume they will use the same unit 
                min_value_with_unit = normal_skill["min_value"]
                # print(f"min_value_with_unit: {min_value_with_unit}")
                min_value, min_unit = self._get_value_and_unit(min_value_with_unit)
                max_value_with_unit = normal_skill["max_value"]
                max_value, max_unit = self._get_value_and_unit(max_value_with_unit)
                
                suggested_values_with_unit = normal_skill["suggested_values"]
                suggested_values_with_unit_tuples = [self._get_value_and_unit(value_with_unit) for value_with_unit in suggested_values_with_unit]

            with open(os.path.join(self.special_skill_path, file_name), 'r') as json_file:
                special_skill = json.load(json_file)
                is_special = False
                special_knob = special_skill["special_knob"]
                if type(special_knob) == str and special_knob.lower() == 'true' or special_knob is True:
                    is_special = True
                    # print(f"special_value: {special_skill['special_value']}")
                    # print(type(special_skill['special_value']))
                    special_value = eval(str(special_skill["special_value"]))

            knob_name = file_name.replace('.json', '')
            # print(f"{knob_name} min_value_with_unit: {min_value_with_unit}, max_value_with_unit: {max_value_with_unit}, unit: {unit}, suggested_values_with_unit: {suggested_values_with_unit}, is_special: {is_special}, special_value: {special_value if is_special else None}")
            # print(f"{knob_name} min_value: {min_value}, max_value: {max_value}, unit: {unit}, suggested_values: {suggested_values}, is_special: {is_special}, special_value: {special_value if is_special else None}")
            self.suggest_knob_info[knob_name] = {
                "min_value": min_value,
                "max_value": max_value,
                "min_unit": min_unit,
                "max_unit": max_unit,
                "suggested_values": suggested_values_with_unit_tuples,
                "is_special": is_special,
                "special_value": special_value if is_special else None
            }
            # if knob_name == 'checkpoint_flush_after':
            #     exit()

    def get_sequence_from_coarse(self, knob):
        info = self.dbms.knob_info[knob]
        if info is None:
            self.log(f"knob {knob} is removed since it does not be found in system_view")
            self.target_knobs.remove(knob) # this knob is not by the DBMS under specific version
            return []

        knob_type = info["vartype"] 
        if knob_type == "enum" or knob_type == "bool":
            knob = self.get_default_space(knob, info)
            self.search_space.add_hyperparameter(knob)
            return []
        
        suggest_info = self.suggest_knob_info.get(knob, None)
        if suggest_info is None:
            return []

        suggested_values = suggest_info["suggested_values"]
        boot_value = info["reset_val"]
        unit = info["unit"]
        knob_type = info["vartype"]

        # hardware constraint if exists
        min_from_sys, max_from_sys = False, False
        min_value = suggest_info["min_value"]
        if min_value is None:
            min_value = info["min_val"]
            min_from_sys = True
        
        max_value = suggest_info["max_value"]
        if max_value is None:
            max_value = info["max_val"]
            max_from_sys = True

        # unify the number based on the unit, then convert the data type(int, float)
        min_value = self._type_transfer(knob_type, unify_unit(min_value, suggest_info["min_unit"]))
        max_value = self._type_transfer(knob_type, unify_unit(max_value, suggest_info["max_unit"]))
        boot_value = self._type_transfer(knob_type, unify_unit(boot_value, unit))
        suggested_values = [self._type_transfer(knob_type, unify_unit(value, unit)) for value, unit in suggested_values]
        # print(f"Coarse sequence for knob {knob}: {min_value}, {max_value}, {boot_value}, suggested_values: {suggested_values}")
        sequence = []
        if boot_value > sys.maxsize / 10:
            boot_value = sys.maxsize / 10

        min_value = min(min_value, boot_value)
        max_value = max(max_value, boot_value)

        # print(f"Coarse sequence for knob {knob}: {min_value}, {max_value}, {boot_value}, suggested_values: {suggested_values}")
        for value in suggested_values:
            if value > max_value or value < min_value:
                self.error_log.warning(f"Suggested value '{value}' is outside of suggested min/max or default range. It is discarded.")
                continue
            for factor in self.factors:
                explore_up = value + factor * (max_value - value) # scale up the suggested value
                explore_down = value + factor * (min_value - value) # scale down the suggested value
                if explore_up < sys.maxsize / 10 and explore_down < explore_up:
                    sequence.append(explore_up)
                    sequence.append(explore_down)
        # print(f"Coarse sequence for knob {knob} after adding scaled values: {sequence}")
        # if a suggested value is not given but a min_val or masx_val is suggested in skill library, equidistant sample.
        if sequence == [] and (not min_from_sys or not max_from_sys):
            for factor in [0.25, 0.5, 0.75]:
                sequence.append(boot_value + factor * (max_value - boot_value)) 
            if not min_from_sys:
                sequence.append(min_value)
            if not max_from_sys:
                sequence.append(max_value)
        sequence.append(boot_value)
        print(f"Coarse sequence for knob {knob} after adding equidistant values: {sequence}")
        if knob_type == "integer":
            sequence = [int(round(value)) for value in sequence]
        else:
            sequence = [float(value) for value in sequence]
        sequence = list(set(sequence)) # remove the duplicated value
        sequence.sort()
        # print(f"Coarse sequence for knob {knob}: {sequence}")
        # if knob == 'checkpoint_flush_after':
        #     exit()
        return sequence