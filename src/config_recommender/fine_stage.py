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
)


class FineStage(FineSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, seed, enhanced, coarse_folder_name):
        super().__init__(dbms, test, timeout, target_knobs_path, seed, enhanced, coarse_folder_name)

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
                print("value", value)
                hp = self.search_space[key]
                if isinstance(hp, CategoricalHyperparameter):
                    transfer_config_value_dict[key] = str(value)
                elif isinstance(hp, UniformIntegerHyperparameter):
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    transfer_config_value_dict[key] = int(value) 
                elif isinstance(hp, UniformFloatHyperparameter):
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    transfer_config_value_dict[key] = float(value)
                else:
                    sys_unit = self.dbms.knob_info[key]["unit"]
                    value = self._transfer_unit_involved(value, sys_unit)
                    transfer_config_value_dict[key] = value
                print("value after transfer", value)
            
            config = Configuration(self.search_space, transfer_config_value_dict)
            configs.insert(0, config) # assume the original costs is ascending order
            config_costs.insert(0, -performance)
        
        print("config_costs", config_costs)
        for _id, config in enumerate(configs):
            smac.runhistory.add(config, config_costs[_id], seed=self.seed)

        smac.optimize()

