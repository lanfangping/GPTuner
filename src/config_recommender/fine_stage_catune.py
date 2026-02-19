from abc import ABC, abstractmethod
from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
from dbms.postgres import PgDBMS
from space_optimizer.fine_space import FineSpace
import json, os, sys, re
from smac import HyperparameterOptimizationFacade, Scenario, initial_design, intensifier
from ConfigSpace import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    Configuration,
    Constant,
    EqualsCondition,
)

# Package from CATune
from optimizer.topo_latin_hypercube_design import TopoLatinHypercubeInitialDesign
from run_SMAC import setup_rules, rule_v5
from search_space.knowledge_based_space import build_search_space, attach_sampler_to_configspace, unify_unit
from sampler.topological_sampler import TopoSampler

class FineStageCATune(FineSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log, rules=[]):
        super().__init__(dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log)
        self.rules = rules

    def optimize(self, name, trials_number, strategy='adaptive'):
        topo_sampler = TopoSampler(
                cs=self.search_space, 
                constraints=self.rules, 
                conditional_activations=self.conditional_activations, 
                seed=self.seed,
                strategy=strategy
            )
        self.search_space = setup_rules(self.search_space, self.rules, self.all_knob_info)
        attach_sampler_to_configspace(cs=self.search_space, sampler=topo_sampler, method_name='sample_configuration')
        scenario = Scenario(
            configspace=self.search_space,
            name = name,
            deterministic=True,
            n_trials=trials_number,
            seed=self.seed,
        )
        init_design = initial_design.DefaultInitialDesign(
            scenario,
            seed=self.seed
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
            config = Configuration(self.search_space, transfer_config_value_dict)
            smac.runhistory.add(config, config_cost, seed=self.seed)
        smac.optimize()


    def define_search_space(self):
        for knob in self.target_knobs:
            info = self.dbms.knob_info[knob]
            if info is None:
                self.target_knobs.remove(knob) # this knob is not by the DBMS under specific version
                continue

            knob_type = info["vartype"] 
            if knob_type == "enum" or knob_type == "bool":
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
                continue
            
            file_name = f"{knob}.json"
            if file_name in os.listdir(self.skill_path):
                with open(os.path.join(self.skill_path, file_name), 'r') as json_file:
                    data = json.load(json_file)
                
                print(f"Defining fine search space for knob: {knob}")
                suggested_values = data["suggested_values"]
                boot_value = info["reset_val"]
                unit = info["unit"]

                # hardware constraint if exists
                min_from_sys, max_from_sys = False, False
                min_value = data["min_value"]
                if min_value is None:
                    min_value = info["min_val"]
                    min_from_sys = True
                
                max_value = data["max_value"]
                if max_value is None:
                    max_value = info["max_val"]
                    max_from_sys = True

                if not min_from_sys:
                    if unit:
                        unit = self._transfer_unit(unit)
                        min_value = self._transfer_unit(min_value) / unit

                        min_value = self._type_transfer(knob_type, min_value)
                        sys_min_value = self._type_transfer(knob_type, info["min_val"])

                        if min_value < sys_min_value:
                            min_value = sys_min_value
                    else:
                        min_value = self._transfer_unit(min_value)

                if not max_from_sys:
                    if unit:
                        unit = self._transfer_unit(unit)
                        max_value = self._transfer_unit(max_value) / unit

                        max_value = self._type_transfer(knob_type, max_value)
                        sys_max_value = self._type_transfer(knob_type, info["max_val"])
                        if max_value > sys_max_value:
                            max_value = sys_max_value
                    else:
                        max_value = self._transfer_unit(max_value)

                # Since the upper bound of some knob in mysql is too big, use GPT's offered upperbound for mysql
                if isinstance(self.dbms, MysqlDBMS):
                    if max_from_sys or max_value >= sys.maxsize / 10:  
                        max_path = "./knowledge_collection/mysql/structured_knowledge/max"
                        with open(os.path.join(max_path, knob+".txt"), 'r') as file:
                            upperbound = file.read()
                        if upperbound != 'null':
                            upperbound = self._type_transfer(knob_type, upperbound)
                            max_value = self._type_transfer(knob_type, max_value)
                            if int(upperbound) < max_value:
                                max_value = upperbound

                # unit transformation
                if unit is not None:
                    unit = self._transfer_unit(unit)
                    suggested_values = [(self._transfer_unit(value) / unit) for value in suggested_values]
                else:
                    suggested_values = [(self._transfer_unit(value)) for value in suggested_values]
                
                # type transformation
                try:
                    suggested_values = [self._type_transfer(knob_type, value) for value in suggested_values]
                    min_value = self._type_transfer(knob_type, min_value)
                    max_value = self._type_transfer(knob_type, max_value)
                    boot_value = self._type_transfer(knob_type, boot_value)
                except:

                    def match_num(value):
                        pattern = r'(\d+(?:\.\d+)?)[\s]*([a-zA-Z]+)'
                        match = re.match(pattern, value)
                        if not match:
                            pattern = r"(\d+(?:\.\d+)?)"  
                            match = re.match(pattern, value)
                            if match:
                                return match.group(1)
                            else:
                                return ""
                        else:
                            return self._transfer_unit(value)

                    # if the knob's unit is null, then if the suggested value contains string, e.g., '150 million', convert it
                    temp = []
                    for value in suggested_values:
                        value = match_num(value)
                        if value != "":
                            temp.append(self._type_transfer(knob_type, value))
                    suggested_values = temp
                    min_value = self._type_transfer(knob_type, match_num(min_value))
                    max_value = self._type_transfer(knob_type, match_num(max_value))
                    boot_value = self._type_transfer(knob_type, match_num(boot_value))
                    
                if boot_value > sys.maxsize / 10:
                    boot_value = sys.maxsize / 10

                # the search space of fine-grained stage should be superset of that of coarse stage
                coarse_sequence = []
                if boot_value > sys.maxsize / 10:
                    boot_value = sys.maxsize / 10


                min_value = min(min_value, boot_value)
                max_value = max(max_value, boot_value)
                # scale up and down the suggested value
                for value in suggested_values:
                    # if knob == 'vacuum_freeze_min_age':
                    #     print(f"suggest value: {value}, min_value: {min_value}, max_value: {max_value}, value? > max_value or value ?< min_value: {value > max_value or value < min_value}")
                    for factor in self.factors:
                        explore_up = value + factor * (max_value - value)
                        explore_down = value + factor * (min_value - value)
                        if explore_up < sys.maxsize / 10 and explore_down < explore_up:
                            coarse_sequence.append(explore_up)
                            coarse_sequence.append(explore_down)
                # if knob == 'vacuum_freeze_min_age':
                #     print("coarse sequence:", coarse_sequence)
                #     exit()
                if coarse_sequence == [] and (not min_from_sys or not max_from_sys):
                    for factor in [0.25, 0.5, 0.75]:
                        coarse_sequence.append(boot_value + factor * (max_value - boot_value)) 
                    if not min_from_sys:
                        coarse_sequence.append(min_value)
                    if not max_from_sys:
                        coarse_sequence.append(max_value)
                coarse_sequence.append(boot_value)

                if max_value > sys.maxsize / 10:
                    max_value = sys.maxsize / 10
                
                if min_value > sys.maxsize / 10:
                    min_value = sys.maxsize / 10

                coarse_sequence = [value for value in coarse_sequence if value < sys.maxsize / 10]
                
                # special_skill_path = f"./knowledge_collection/{self.dbms.name}/structured_knowledge/special/"
                # check if this knob is special knob
                special = False
                special_value = None
                if file_name in os.listdir(self.special_skill_path):
                    with open(os.path.join(self.special_skill_path, file_name), 'r') as json_file:
                        special_skill = json.load(json_file)
                        special_knob = special_skill["special_knob"]
                        if type(special_knob) == str and special_knob.lower() == 'true' or special_knob is True:
                            special = True
                            special_value = special_skill["special_value"]
                        # if special is True:
                        #     special_value = special_skill["special_value"]
                
                if knob_type == "integer":  
                    coarse_sequence = [int(value) for value in coarse_sequence]
                    self.log.info(f"Coarse sequence for knob {knob}: {coarse_sequence}")
                    coarse_sequence = [unify_unit(value, info['unit']) for value in coarse_sequence]
                    self.log.info(f"Unified coarse sequence for knob {knob}: {coarse_sequence}")
                    min_value = min(min_value, min(coarse_sequence))
                    max_value = max(max_value, max(coarse_sequence))
                    self.log.info(f"min_value: {min_value}, max_value: {max_value} for knob {knob}")
                    min_value = unify_unit(min_value, info['unit'])
                    max_value = unify_unit(max_value, info['unit'])
                    self.log.info(f"unified min_value: {min_value}, unified max_value: {max_value} for knob {knob}")
                    if min_value == max_value: # the parameter has the equal min_value and max_value
                        normal_para = Constant(knob, int(max_value))
                    elif min_value > max_value:
                        print(f"error: min_value is lager than max_value: {min_value} > {max_value}, skip")
                        continue
                    else:   
                        normal_para = UniformIntegerHyperparameter(
                            knob, 
                            int(min_value), 
                            int(max_value),
                            default_value = int(boot_value),
                        )

                    if special:
                        control_para = CategoricalHyperparameter(f"control_{knob}", ["0", "1"], default_value="0") 
                        if type(special_value) is list:
                            # special_para = OrdinalHyperparameter(f"special_{knob}", [int(value) for value in special_value])
                            special_para = CategoricalHyperparameter(f"special_{knob}", [str(value) for value in special_value])
                        else:
                            special_para = Constant(f"special_{knob}", int(special_value))

                        self.search_space.add_hyperparameters([control_para, normal_para, special_para])
                        
                        normal_cond = EqualsCondition(self.search_space[knob], self.search_space[f"control_{knob}"], "0")
                        special_cond = EqualsCondition(self.search_space[f"special_{knob}"], self.search_space[f"control_{knob}"], "1")
                        
                        self.search_space.add_conditions([normal_cond, special_cond])
                    else:
                        self.search_space.add_hyperparameter(normal_para)
                    
                elif knob_type == "real":
                    coarse_sequence = [float(value) for value in coarse_sequence]
                    self.log.info(f"Coarse sequence for knob {knob}: {coarse_sequence}")
                    coarse_sequence = [unify_unit(value, info['unit']) for value in coarse_sequence]
                    self.log.info(f"Unified coarse sequence for knob {knob}: {coarse_sequence}")

                    min_value = min(min_value, min(coarse_sequence))
                    max_value = max(max_value, max(coarse_sequence))
                    self.log.info(f"min_value: {min_value}, max_value: {max_value} for knob {knob}")
                    min_value = unify_unit(min_value, info['unit'])
                    max_value = unify_unit(max_value, info['unit'])
                    self.log.info(f"unified min_value: {min_value}, unified max_value: {max_value} for knob {knob}")

                    normal_para = UniformFloatHyperparameter(
                        knob,
                        float(min_value),
                        float(max_value),
                        default_value = float(boot_value),
                    )
                    if special:
                        control_para = CategoricalHyperparameter(f"control_{knob}", ["0", "1"], default_value="0") 
                        if type(special_value) is list:
                            special_para = CategoricalHyperparameter(f"special_{knob}", [str(value) for value in special_value])
                        else:
                            special_para = Constant(f"special_{knob}", float(special_value))

                        self.search_space.add_hyperparameters([control_para, normal_para, special_para])
                        normal_cond = EqualsCondition(self.search_space[knob], self.search_space[f"control_{knob}"], "0")
                        special_cond = EqualsCondition(self.search_space[f"special_{knob}"], self.search_space[f"control_{knob}"], "1")
                        
                        self.search_space.add_conditions([normal_cond, special_cond])
                    else:
                        self.search_space.add_hyperparameter(normal_para)
            else:
                info = self.dbms.knob_info[knob]
                if info is None:
                    continue
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
    
    def get_default_space(self, knob_name, info):
        boot_value = info["reset_val"]
        min_value = info["min_val"]
        max_value = info["max_val"]
        knob_type = info["vartype"]
        self.log.info(f"Boot value: {boot_value}, min value: {min_value}, max value: {max_value} for knob {knob_name}")
        boot_value = unify_unit(boot_value, info["unit"])
        min_value = unify_unit(min_value, info["unit"])
        max_value = unify_unit(max_value, info["unit"])
        self.log.info(f"Unified boot value: {boot_value}, unified min value: {min_value}, unified max value: {max_value} for knob {knob_name}")

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