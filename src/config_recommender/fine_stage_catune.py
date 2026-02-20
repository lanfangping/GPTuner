from abc import ABC, abstractmethod
from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
from dbms.postgres import PgDBMS
from space_optimizer.fine_space_catune import FineSpaceCATune
import json, os, sys, re, time
from smac import HyperparameterOptimizationFacade, Scenario, initial_design, intensifier
from ConfigSpace import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    Constant,
    EqualsCondition,
)
from smac.runhistory.dataclasses import TrialValue, TrialInfo
import copy, random

# Package from CATune
from optimizer.topo_latin_hypercube_design import TopoLatinHypercubeInitialDesign
from run_SMAC import setup_rules, rule_v5
from search_space.knowledge_based_space import build_search_space, attach_sampler_to_configspace, unify_unit
from sampler.topological_sampler import TopoSampler

class FineStageCATune(FineSpaceCATune):

    def __init__(self, dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log, rules=[], conditional_activations=[]):
        super().__init__(dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log)
        self.rules = rules
        self.conditional_activations += conditional_activations
        knob_info_path = os.path.join(os.path.dirname(os.path.dirname(special_skill_path)), 'knob_info/system_view.json')
        self.all_knob_info = {}
        for knob, knob_info in json.load(open(knob_info_path, "r")).items():
            knob_info['default'] = knob_info['boot_val']
            self.all_knob_info[knob] = knob_info

        self.original_config_file = os.path.join(results_folder, f"{self.dbms.name}/fine_rules/{self.seed}/original_configs.json")
        self.applied_config_file = os.path.join(results_folder, f"{self.dbms.name}/fine_rules/{self.seed}/applied_configs.json")
        self.feasible_config_file = os.path.join(results_folder, f"{self.dbms.name}/fine_rules/{self.seed}/feasible_configs.json")

    def optimize(self, name, trials_number, strategy='adaptive'):
        topo_sampler = TopoSampler(
                cs=self.search_space, 
                constraints=self.rules, 
                conditional_activations=self.conditional_activations, 
                seed=self.seed,
                strategy=strategy
            )
        search_space_without_rules = copy.deepcopy(self.search_space)
        self.search_space = setup_rules(self.search_space, self.rules, self.all_knob_info)
        attach_sampler_to_configspace(cs=self.search_space, sampler=topo_sampler, method_name='sample_configuration')
        scenario = Scenario(
            configspace=self.search_space,
            name = name,
            deterministic=True,
            n_trials=trials_number,
            seed=self.seed,
        )
        
        initial_configurations = []
        initial_costs = []
        # how to be guided by coarse-grained tuning
        with open(self.coarse_path, "r") as json_file:
            data = json.load(json_file)
        costs = []
        
        for i in range(30):
            try:
                cost_item = data["data"][i][4]
            except:
                cost_item = data["data"][i]['cost']
            costs.append(cost_item)
        # the [:x] configurations with minimal costs
        index_min_pairs = sorted(enumerate(costs), key=lambda x: x[1])[:30]
        # no ordering
        # for index, value in enumerate(costs):
        for index, value in index_min_pairs:
            config_id = index + 1
            config_value_dict = data["configs"][str(config_id)]
            try:
                config_cost = data["data"][index][4]
            except:
                config_cost = data["data"][index]['cost']
            assert value == config_cost
            # make type transformation from coarse to fine 
            transfer_config_value_dict = {}
            for key, value in config_value_dict.items():
                # self.error_log.info(f"Processing config item: {key} with value: {value}")
                if key.startswith("control_") or key.startswith("special_"):
                    transfer_config_value_dict[key] = value
                    continue
                hp = self.search_space[key]
                if isinstance(hp, CategoricalHyperparameter):
                    transfer_config_value_dict[key] = value
                elif isinstance(hp, UniformIntegerHyperparameter):
                    transfer_config_value_dict[key] = int(value) 
                elif isinstance(hp, UniformFloatHyperparameter):
                    transfer_config_value_dict[key] = float(value)
                else:
                    transfer_config_value_dict[key] = eval(value)
            
            # the coarse config may break the rules, so we use the search space without rules to construct the configuration
            config = Configuration(search_space_without_rules, transfer_config_value_dict) 
            config.origin = "Initial Design: coarse tuning"
            initial_configurations.append(config)
            initial_costs.append(config_cost)
            # smac.runhistory.add(config, config_cost, seed=self.seed)

        # init_design = initial_design.DefaultInitialDesign(
        #     scenario,
        #     seed=self.seed
        # )
        # Modify the initial design to use our custom initial design
        initial_design=HyperparameterOptimizationFacade.get_initial_design(
            scenario, 
            n_configs=0,  # Do not use the default initial design
            additional_configs=initial_configurations  # Use the configurations previously evaluated as initial design
                                            # This only passes the configurations but not the cost!
        )
        # smac = HyperparameterOptimizationFacade(
        #     scenario=scenario,
        #     initial_design=initial_design,
        #     target_function=self.set_and_replay,
        #     intensifier=intensifier.Intensifier(scenario, retries=30),
        # )
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            initial_design=initial_design,
            target_function=self.set_and_replay, # tuning process with normal random optimization
            overwrite=False,
        )

        # Convert previously evaluated configurations into TrialInfo and TrialValue instances to pass to SMAC
        trial_infos = [TrialInfo(config=c, seed=self.seed) for c in initial_configurations]
        trial_values = [TrialValue(cost=c) for c in costs]

        # Warmstart SMAC with the trial information and values
        for info, value in zip(trial_infos, trial_values):
            smac.tell(info, value, save=True)
        smac.optimize()


    def convert_config(self, config):
        convert_config = {}
        for knob, value in config.items():
            if knob.startswith("control_") or knob.startswith("special_"):
                convert_config[knob] = value
                continue
            unit = self.all_knob_info[knob].get('unit', None)
            convert_value = unify_unit(value, 'kB', unit)
            convert_config[knob] = convert_value
        return convert_config

    
    def set_and_replay(self, config, seed=0):
        self._log_original_config(dict(config))
        config = self.convert_config(config=config)
        begin_time = time.time()
        cost = self.set_and_replay_ori(config, seed)
        end_time = time.time()
        self._log(begin_time, end_time)
        return cost
    

    def _log_original_config(self, config):
        if os.path.exists(self.original_config_file):
            all_original_config = json.load(open(self.original_config_file, 'r'))
            all_original_config[f"{self.round}"] = config
            json.dump(all_original_config, open(self.original_config_file, 'w'), indent=4)
        else:
            with open(self.original_config_file, 'w') as f:
                json.dump({
                    f"{self.round}": config
                }, f, indent=4)

    def _log_applied_config(self, config):
        if os.path.exists(self.applied_config_file):
            all_applied_config = json.load(open(self.applied_config_file, 'r'))
            all_applied_config[f"{self.round}"] = config
            json.dump(all_applied_config, open(self.applied_config_file, 'w'), indent=4)
        else:
            with open(self.applied_config_file, 'w') as f:
                json.dump({
                    f"{self.round}": config
                }, f, indent=4)
    
    def _log_feasible_config(self, feasible:bool):
        if os.path.exists(self.feasible_config_file):
            all_feasible_config = json.load(open(self.feasible_config_file, 'r'))
            all_feasible_config[f"{self.round}"] = feasible
            json.dump(all_feasible_config, open(self.feasible_config_file, 'w'), indent=4)
        else:
            with open(self.feasible_config_file, 'w') as f:
                json.dump({
                    f"{self.round}": feasible
                }, f, indent=4)