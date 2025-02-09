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
import functools
from dbms.mysql import MysqlDBMS
from ConfigSpace import (
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
)


class DefaultSpace:
    """ Base template of GPTuner"""
    def __init__(self, dbms, test, timeout, target_knobs_path, log, seed=1, enhanced=False, enhanced_starting_path='"../DBtuningDataset/historical_best_config/historical_best_tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202412032307.json"', folder_name='optimization_results'):
        self.dbms = dbms
        self.seed = seed if seed is not None else 1
        self.test = test # workload type
        self.timeout = timeout
        self.target_knobs_path = target_knobs_path
        self.enhanced_starting_path = enhanced_starting_path
        # if mode=='past_best':
        #     self.enhanced_starting_path = f"../DBtuningDataset/historical_best_config/historical_best_tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202412032307.json"
        # elif mode == 'all_ordered':
        #     self.enhanced_starting_path = f"../DBtuningDataset/historical_ordered_config/historical_ordered_tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202412032307.json"
        self.round = 0
        self.summary_path = f"./experiments_results/{folder_name}/temp_results"
        self.benchmark_copy_db = ['tpcc', 'twitter', "sibench", "voter", "tatp", "smallbank", "seats"]   # Some benchmark will insert or delete data, Need to be rewrite each time.
        self.benchmark_latency = ['tpch']
        self.search_space = ConfigurationSpace()
        self.skill_path = f"./knowledge_collection/{self.dbms.name}/structured_knowledge/normal"
        self.logger = log
        self.target_knobs = self.knob_select(enhanced)
        self.logger.info(f"the number of knobs: {len(self.target_knobs)}")
        self.logger.info(f"target knobs: {self.target_knobs}")
        if self.test in self.benchmark_copy_db:
            self.dbms.copy_db(target_db=f"{self.test}_template", source_db="benchbase")
        # self.penalty = 0
        self.penalty = self.get_default_result()
        print(f"DEFAULT : {self.penalty}")
        self.log_file = f"./experiments_results/{folder_name}/{self.dbms.name}/log/{self.seed}_log.txt"
        self.init_log_file()
        self.prev_end = 0


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
        pattern = r'(\d+\.\d+|\d+)([a-zA-Z]+)'
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
            'day': 24 * 60 * 60000,
            'd': 24 * 60 * 60000,
        }
        if unit not in unit_to_size.keys():
            print(f"Illegal unit:{unit}")
            return float(number)
        return float(number) * unit_to_size[unit]
    
    def _transfer_unit_involved(self, value, sys_unit):
        value = str(value)
        value = value.replace(" ", "")
        value = value.replace(",", "")
        if value.isalpha():
            value = "1" + value
        pattern = r'(\d+\.\d+|\d+)([a-zA-Z]+)'
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
            'day': 24 * 60 * 60000,
            'd': 24 * 60 * 60000,
            '8kB': 8 * 1e3
        }
        if unit not in unit_to_size.keys():
            print(f"Illegal unit:{unit}")
            return float(number)
        if sys_unit in unit_to_size.keys():
            return float(number) * unit_to_size[unit] / unit_to_size[sys_unit]
        return float(number) * unit_to_size[unit]
    
    def _type_transfer(self, knob_type, value):
        value = str(value)
        value = value.replace(",", "")
        if knob_type == "integer":
            return int(round(float(value)))
        if knob_type == "real":
            return float(value)

    def knob_select(self, enhanced):
        """ 
            Select which knobs to be tuned, store the names in 'self.target_knobs' 
            Default implementation is to use fixed knobs. Provide the path to the file containing the knobs' names.
        """
        target_knobs = []
        if enhanced:
            # print("enhanced parameters")
            self.logger.info("knob selected by enhanced good startings")
            enhanced_starting_data = json.load(open(self.enhanced_starting_path, 'r'))
            for _id, data in enhanced_starting_data.items():
                config = data['config']
                if config == 'default settings':
                    continue
                
                for knob, value in config.items():
                    if knob not in self.dbms.knob_info.keys() or "vartype" not in self.dbms.knob_info[knob] or self.dbms.knob_info[knob]["vartype"] == "string":
                        continue

                    if knob not in target_knobs:
                        target_knobs.append(knob)
        else:
            self.logger.info(f"knob is pre-selected from {self.target_knobs_path}")
            current_directory = os.getcwd()
            print(current_directory)
            with open(self.target_knobs_path, 'r') as file:
                lines = file.readlines()
            candidate_knobs = [line.strip() for line in lines]
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
        begin_time = time.time()
        cost = self.set_and_replay_ori(config, seed)
        end_time = time.time()
        self._log(begin_time, end_time)
        return cost


    def set_and_replay_ori(self, config, seed=0):
        self.round += 1
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
            dbms.set_knob(knob, value)
            
        dbms.reconfigure()
        if self.test not in self.benchmark_latency:
            if dbms.failed_times == 4:
                return -int(self.penalty) / 2
        else:
            if dbms.failed_times == 4:
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
            # update worst_perf
            if self.test not in self.benchmark_latency:
                return -int(self.penalty) / 2
                ###tpch
            else:
                return self.penalty * 2
    

        if self.test not in self.benchmark_latency:
            return -throughput
        else:
            return average_latency
        

    @abstractmethod
    def define_search_space(self):
        pass


    def define_search_space_default(self):
        for knob in self.target_knobs:
            print(knob)
            info = self.dbms.knob_info[knob]
            if info is None:
                print(f"{knob} is not by the DBMS under specific version")
                self.target_knobs.remove(knob) # this knob is not by the DBMS under specific version
                continue

            knob = self.get_default_space(knob, info)
            self.search_space.add_hyperparameter(knob)