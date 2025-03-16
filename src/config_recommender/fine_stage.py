from abc import ABC, abstractmethod
from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
# from dbms.postgres import PgDBMS
from dbms.postgres_docker import PgDBMS
from space_optimizer.fine_space import FineSpace
import json
from smac import HyperparameterOptimizationFacade, Scenario, initial_design, intensifier
from ConfigSpace import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    Constant,
)
from ConfigSpace.util import deactivate_inactive_hyperparameters
import os


class FineStage(FineSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, log, seed, enhanced, enhanced_starting_path, enhanced_strategy, fine, coarse_folder_name):
        super().__init__(dbms, test, timeout, target_knobs_path, log, seed, enhanced, enhanced_starting_path, enhanced_strategy, fine, coarse_folder_name)

    def optimize(self, name, trials_number):
        scenario = Scenario(
            configspace=self.search_space,
            name = name,
            deterministic=True,
            n_trials=trials_number,
            seed=self.seed,
        )
        init_design = initial_design.DefaultInitialDesign(
            scenario,
        )
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            initial_design=init_design,
            target_function=self.set_and_replay,
            intensifier=intensifier.Intensifier(scenario, retries=30),
        )
        # how to be guided by coarse-grained tuning
        with open(self.coarse_path, "r") as json_file:
            data = json.load(json_file)
        costs = []
        for i in range(30):
            costs.append(data["data"][i][4])
        # the [:x] configurations with minimal costs
        index_min_pairs = sorted(enumerate(costs), key=lambda x: x[1])[:30]
        # no ordering
        # for index, value in enumerate(costs):
        for index, value in index_min_pairs:
            config_id = index + 1
            config_value_dict = data["configs"][str(config_id)]
            config_cost = data["data"][index][4]
            assert value == config_cost
            # make type transformation from coarse to fine 
            transfer_config_value_dict = {}
            for key, value in config_value_dict.items():
                if key.startswith("control_") or key.startswith("special_"):
                    transfer_config_value_dict[key] = value
                    continue
                hp = self.search_space[key]
                if isinstance(hp, CategoricalHyperparameter):
                    transfer_config_value_dict[key] = str(value)
                elif isinstance(hp, UniformIntegerHyperparameter):
                    transfer_config_value_dict[key] = int(value) 
                elif isinstance(hp, UniformFloatHyperparameter):
                    transfer_config_value_dict[key] = float(value)
                else:
                    transfer_config_value_dict[key] = value
            config = Configuration(self.search_space, transfer_config_value_dict)
            smac.runhistory.add(config, config_cost, seed=self.seed)
        smac.optimize()

    def optimize_enhanced_starting(self, name, trials_number):
        scenario = Scenario(
            configspace=self.search_space,
            name = name,
            deterministic=True,
            n_trials=trials_number,
            seed=self.seed,
        )
        init_design = initial_design.DefaultInitialDesign(
            scenario,
        )   

        with open(self.enhanced_starting_path, "r") as json_file:
            enhanced_starting_data = json.load(json_file)
        contain_default_settings = False
        for _id, data in enhanced_starting_data.items():
            config = data['config']
            if config == 'default settings':
                contain_default_settings = True
                break
        
        enhanced_starting_configs_number = len(enhanced_starting_data)
        if contain_default_settings:
            enhanced_starting_configs_number -= 1

        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            initial_design=init_design,
            target_function=self.set_and_replay,
            intensifier=intensifier.Intensifier(scenario, retries=enhanced_starting_configs_number),
        )

        configs = []
        config_costs = []
        for _id, data in enhanced_starting_data.items():
            config_value_dict = data['config']
            if config_value_dict == 'default settings':
                contain_default_settings = True
                continue

            performance = data["performance"]

            # make type transformation from coarse to fine 
            transfer_config_value_dict = {}
            for key, value in config_value_dict.items():
                if key not in self.dbms.knob_info.keys() or "vartype" not in self.dbms.knob_info[key] or self.dbms.knob_info[key]["vartype"] == "string":
                    continue
                
                hp = self.search_space[key]
                # print(f'{key}:{value} - {hp.is_legal(value)}')
                if isinstance(hp, Constant):
                    transfer_config_value_dict[key] = hp.value
                elif isinstance(hp, CategoricalHyperparameter):
                    transfer_config_value_dict[key] = str(value)
                elif isinstance(hp, UniformIntegerHyperparameter):
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    if value >= hp.upper:
                        transfer_config_value_dict[key] = hp.upper
                    elif value <= hp.lower:
                        transfer_config_value_dict[key] = hp.lower
                    else:
                        transfer_config_value_dict[key] = int(value) 

                elif isinstance(hp, UniformFloatHyperparameter):
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    if value >= hp.upper:
                        transfer_config_value_dict[key] = hp.upper
                    elif value <= hp.lower:
                        transfer_config_value_dict[key] = hp.lower
                    else:
                        transfer_config_value_dict[key] = float(value)
                else:   
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    if sys_unit:
                        value = self._transfer_unit_involved(value, sys_unit)
                    transfer_config_value_dict[key] = value
                print("value after transfer", value)

            uncovered_knobs = list(set(self.target_knobs).difference(set(config_value_dict.keys())))
            print("\n")
            for knob in uncovered_knobs:
                print("uncovered knob:", knob)            
                hp = self.search_space[knob]
                transfer_config_value_dict[knob] = hp.default_value

            config = Configuration(self.search_space, transfer_config_value_dict)
            configs.insert(0, config) # assume the original costs is ascending order
            config_costs.insert(0, -performance)
        
        for _id, config in enumerate(configs):
            smac.runhistory.add(config, config_costs[_id], seed=self.seed)

        smac.optimize()

    def optimize_enhanced_starting_wellsearchspace(self, name, trials_number):
        scenario = Scenario(
            configspace=self.search_space,
            name = name,
            deterministic=True,
            n_trials=trials_number,
            seed=self.seed,
        )
        init_design = initial_design.DefaultInitialDesign(
            scenario,
        )   

        with open(self.enhanced_starting_path, "r") as json_file:
            enhanced_starting_data = json.load(json_file)
        contain_default_settings = False
        for _id, data in enhanced_starting_data.items():
            config = data['config']
            if config == 'default settings':
                contain_default_settings = True
                break
        
        enhanced_starting_configs_number = len(enhanced_starting_data)
        if contain_default_settings:
            enhanced_starting_configs_number -= 1

        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            initial_design=init_design,
            target_function=self.set_and_replay,
            intensifier=intensifier.Intensifier(scenario, retries=enhanced_starting_configs_number),
        )
        special_skill_path = f"./knowledge_collection/{self.dbms.name}/structured_knowledge/special/"

        configs = []
        config_costs = []
        i = 0
        for _id, data in enhanced_starting_data.items():
            
            config_value_dict = data['config']
            if config_value_dict == 'default settings':
                contain_default_settings = True
                continue

            performance = data["performance"]

            # make type transformation from coarse to fine 
            transfer_config_value_dict = {}
            for key, value in config_value_dict.items():
                # print("\n", key, value)
                if key not in self.dbms.knob_info.keys() or "vartype" not in self.dbms.knob_info[key] or self.dbms.knob_info[key]["vartype"] == "string":
                    continue

                if key.startswith("control_") or key.startswith("special_"):
                    transfer_config_value_dict[key] = value
                    continue
                
                # set special control parameter for knobs
                file_name = f"{key}.json"
                
                # check if this knob is special knob
                special = False
                special_value = None
                special_skill_path = f"./knowledge_collection/{self.dbms.name}/structured_knowledge/special/"
                if file_name in os.listdir(special_skill_path):
                    with open(os.path.join(special_skill_path, file_name), 'r') as json_file:
                        special_skill = json.load(json_file)
                        special_knob = special_skill["special_knob"]
                        # print(f"{special_knob}, type: {type(special_knob)}")
                        if type(special_knob) == str and special_knob.lower() == 'true' or special_knob is True:
                            special = True
                            special_value = special_skill["special_value"]

                if special is True:
                    if f'special_{key}' in self.search_space.get_hyperparameter_names():
                        hp = self.search_space[f'special_{key}']
                        sys_unit = self.dbms.knob_info[key]["unit"]
                        if sys_unit is not None:
                            value = self._transfer_unit_involved(value, sys_unit)
                        # print(value, hp.value)
                        # print(value == hp.value)
                        if value == hp.value:
                            transfer_config_value_dict[f'control_{key}'] = str(1)
                            transfer_config_value_dict[f'special_{key}'] = hp.value
                        else:
                            transfer_config_value_dict[f'control_{key}'] = str(0)
                        
                        # if key == 'wal_receiver_status_interval':
                        #     print(transfer_config_value_dict)
                        #     exit()
                            # input()
                # input()
                # print(key)
                hp = self.search_space[key]
                # print(f'{key}:{value} - {hp.is_legal(value)}')
                if isinstance(hp, CategoricalHyperparameter):
                    transfer_config_value_dict[key] = str(value)
                elif isinstance(hp, UniformIntegerHyperparameter):
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    if value >= hp.upper:
                        transfer_config_value_dict[key] = hp.upper
                    elif value <= hp.lower:
                        transfer_config_value_dict[key] = hp.lower
                    else:
                        transfer_config_value_dict[key] = int(value) 
                    
                elif isinstance(hp, UniformFloatHyperparameter):
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    if value >= hp.upper:
                        transfer_config_value_dict[key] = hp.upper
                    elif value <= hp.lower:
                        transfer_config_value_dict[key] = hp.lower
                    else:
                        transfer_config_value_dict[key] = float(value)
                else:   
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    transfer_config_value_dict[key] = value
                
                # if key == 'wal_receiver_status_interval':
                #     print(transfer_config_value_dict)
                    # exit()
                # print("value after transfer", value)

                # if key == 'geqo_threshold':
                #     exit()
                # input()
            # print(transfer_config_value_dict)
            # print(self.target_knobs)
            
            uncovered_knobs = list(set(self.target_knobs).difference(set(config_value_dict.keys())))
            print("\n")
            for knob in uncovered_knobs:
                print("uncovered knob:", knob)
                # set special control parameter for knobs
                file_name = f"{knob}.json"
                
                # check if this knob is special knob
                special = False
                if file_name in os.listdir(special_skill_path):
                    with open(os.path.join(special_skill_path, file_name), 'r') as json_file:
                        special_skill = json.load(json_file)

                    special_knob = special_skill["special_knob"]
                    if type(special_knob) == str and special_knob.lower() == 'true' or special_knob is True:
                        special = True

                    if special:
                        transfer_config_value_dict[f'control_{knob}'] = str(0)
                        # print(transfer_config_value_dict[f'control_{knob}'])

                hp = self.search_space[knob]
                transfer_config_value_dict[knob] = hp.default_value

            # hp_wal = self.search_space['wal_receiver_status_interval']
            # print(hp_wal)
            # print(self.search_space.get('wal_receiver_status_interval'))
            # print(self.search_space['wal_receiver_status_interval'].value)
            print(transfer_config_value_dict)
            # print(f"{i}th config\n")
            i += 1
            config = deactivate_inactive_hyperparameters(transfer_config_value_dict, self.search_space)
            # config = Configuration(self.search_space, transfer_config_value_dict)
            # print(config)
            # print(config.get('wal_receiver_status_interval'))
            configs.insert(0, config) # assume the original costs is ascending order
            config_costs.insert(0, -performance)
        
        # print("config_costs", config_costs)
        for _id, config in enumerate(configs):
            smac.runhistory.add(config, config_costs[_id], seed=self.seed)

        smac.optimize()

