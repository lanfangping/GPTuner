from abc import ABC, abstractmethod
from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
# from dbms.postgres import PgDBMS
from dbms.postgres_docker import PgDBMS
import sys
import os
import json
import re
from smac import HyperparameterOptimizationFacade, Scenario, initial_design, intensifier
from ConfigSpace import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter,
    Constant,
    Configuration,
    EqualsCondition,
)
from collections import defaultdict

class FineSpace(DefaultSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, log, seed, enhanced, enhanced_starting_path, enhanced_strategy, fine, coarse_folder_name):
        super().__init__(dbms, test, timeout, target_knobs_path, log, seed, enhanced, enhanced_starting_path, coarse_folder_name)
        self.factors = [0, 0.25, 0.5]
        if enhanced:
            if fine.lower() == 'knowledge':
                self.define_search_space() # if enhanced, then selected knobs from startings
            elif fine.lower() == 'default':
                if enhanced_strategy == 'suggest_values':
                    self.define_enhancedstartings_with_enhancedspace() # enhanced startings + enhanced space(treat as suggest values) + selected knobs from startings 
                elif enhanced_strategy == 'search_range':
                    self.define_enhancedstartings_with_enhancedspace2() # enhanced startings + enhanced space(treat as search range) + selected knobs from startings 
            else:
                log.error(f"invalid arg 'fine' {fine}")
        else:
            if fine.lower() == 'knowledge':
                self.define_search_space() # if enhanced, then pre-select knobs
            elif fine.lower() == 'default':
                self.define_search_space_default() # no knowledge + default space + pre-select knobs
            else:
                log.error(f"invalid arg 'fine' {fine}")
                exit()
        # print(self.search_space)
        log.info(f"Defined search space: {self.search_space}")
        self.coarse_path = f"./experiments_results/{coarse_folder_name}/{self.dbms.name}/coarse/{self.seed}/runhistory.json"
        
    def define_search_space(self):
        """
        pre-selected parameters and well-reduced search space according to suggested values and suggested min/max
        if enhanced is True, self.target_knobs is a collection of parameters in good starting points.
        """
        for knob in self.target_knobs:
            info = self.dbms.knob_info[knob]
            if info is None:
                self.target_knobs.remove(knob) # this knob is not by the DBMS under specific version
                continue
            print(f"Defining fine search space for knob: {knob}")
            file_name = f"{knob}.json"

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
                            
            knob_type = info["vartype"] 
            if knob_type == "enum" or knob_type == "bool":
                normal_para = self.get_default_space(knob, info)
                if special is True:
                    control_para = CategoricalHyperparameter(f"control_{knob}", ["0", "1"], default_value="0") 
                    if type(special_value) is list:
                        # special_para = OrdinalHyperparameter(f"special_{knob}", [int(value) for value in special_value])
                        special_para = CategoricalHyperparameter(f"special_{knob}", [str(value) for value in special_value])
                    else:
                        special_para = Constant(f"special_{knob}", str(special_value))
                    
                    self.search_space.add_hyperparameters([control_para, normal_para, special_para])
                    
                    normal_cond = EqualsCondition(self.search_space[knob], self.search_space[f"control_{knob}"], "0")
                    special_cond = EqualsCondition(self.search_space[f"special_{knob}"], self.search_space[f"control_{knob}"], "1")
                    # print("Defining control knob:", f"control_{knob}")
                    # print("Defining special knob:", f"special_{knob}\n")
                    self.search_space.add_conditions([normal_cond, special_cond])
                else:
                    self.search_space.add_hyperparameter(normal_para)

                continue
            
            if file_name in os.listdir(self.skill_path):
                with open(os.path.join(self.skill_path, file_name), 'r') as json_file:
                    data = json.load(json_file)
                
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

                if not max_from_sys:
                    if unit:
                        unit = self._transfer_unit(unit)
                        max_value = self._transfer_unit(max_value) / unit

                        max_value = self._type_transfer(knob_type, max_value)
                        sys_max_value = self._type_transfer(knob_type, info["max_val"])
                        if max_value > sys_max_value:
                            max_value = sys_max_value
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
                
                # type transformation
                try:
                    suggested_values = [self._type_transfer(knob_type, value) for value in suggested_values]
                    min_value = self._type_transfer(knob_type, min_value)
                    max_value = self._type_transfer(knob_type, max_value)
                    boot_value = self._type_transfer(knob_type, boot_value)
                except:

                    def match_num(value):
                        pattern = r"(\d+)"
                        match = re.match(pattern, value)
                        if match:
                            return match.group(1)
                        else:
                            return ""

                    pattern = r"(\d+)"
                    suggested_values = [self._type_transfer(knob_type, re.match(pattern, value).group(1)) for value in suggested_values if re.match(pattern, value) is not None]
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
                    for factor in self.factors:
                        explore_up = value + factor * (max_value - value)
                        explore_down = value + factor * (min_value - value)
                        if explore_up < sys.maxsize / 10 and explore_down < explore_up:
                            coarse_sequence.append(explore_up)
                            coarse_sequence.append(explore_down)
                
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
                # if file_name in os.listdir(special_skill_path):
                #     with open(os.path.join(special_skill_path, file_name), 'r') as json_file:
                #         special_skill = json.load(json_file)
                #     special = special_skill["special_knob"]
                #     if special is True:
                #         special_value = special_skill["special_value"]
                
                if knob_type == "integer":  
                    coarse_sequence = [int(value) for value in coarse_sequence]
                    min_value = min(min_value, min(coarse_sequence))
                    max_value = max(max_value, max(coarse_sequence))
                    normal_para = UniformIntegerHyperparameter(
                        knob, 
                        int(min_value), 
                        int(max_value),
                        default_value = int(boot_value),
                    )

                    # special = special_skill["special_knob"]
                    if special:
                        # special_value = special_skill["special_value"]
                        # print(special_value)
                        control_para = CategoricalHyperparameter(f"control_{knob}", ["0", "1"], default_value="0") 
                        if type(special_value) is list:
                            # special_para = OrdinalHyperparameter(f"special_{knob}", [int(value) for value in special_value])
                            special_para = CategoricalHyperparameter(f"special_{knob}", [str(value) for value in special_value])
                        else:
                            special_para = Constant(f"special_{knob}", int(special_value))
                        
                        self.search_space.add_hyperparameters([control_para, normal_para, special_para])
                        
                        normal_cond = EqualsCondition(self.search_space[knob], self.search_space[f"control_{knob}"], "0")
                        special_cond = EqualsCondition(self.search_space[f"special_{knob}"], self.search_space[f"control_{knob}"], "1")
                        # print("Defining control knob:", f"control_{knob}")
                        # print("Defining special knob:", f"special_{knob}\n")
                        self.search_space.add_conditions([normal_cond, special_cond])
                    else:
                        self.search_space.add_hyperparameter(normal_para)
                    
                elif knob_type == "real":
                    coarse_sequence = [float(value) for value in coarse_sequence]
                    min_value = min(min_value, min(coarse_sequence))
                    max_value = max(max_value, max(coarse_sequence))
                    normal_para = UniformFloatHyperparameter(
                        knob,
                        float(min_value),
                        float(max_value),
                        default_value = float(boot_value),
                    )
                    # special = special_skill["special_knob"]
                    if special:
                        # special_value = special_skill["special_value"]
                        control_para = CategoricalHyperparameter(f"control_{knob}", ["0", "1"], default_value="0") 
                        if type(special_value) is list:
                            special_para = CategoricalHyperparameter(f"special_{knob}", [str(value) for value in special_value])
                        else:
                            special_para = Constant(f"special_{knob}", float(special_value))

                        self.search_space.add_hyperparameters([control_para, normal_para, special_para])
                        normal_cond = EqualsCondition(self.search_space[knob], self.search_space[f"control_{knob}"], "0")
                        special_cond = EqualsCondition(self.search_space[f"special_{knob}"], self.search_space[f"control_{knob}"], "1")
                        
                        # print("Defining control knob:", f"control_{knob}")
                        # print("Defining special knob:", f"special_{knob}\n")
                        self.search_space.add_conditions([normal_cond, special_cond])
                    else:
                        self.search_space.add_hyperparameter(normal_para)
            else:
                info = self.dbms.knob_info[knob]
                if info is None:
                    continue
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)

    def define_enhancedstartings_with_enhancedspace(self):
        """
        enhanced by feeding good starting configurations 
        but the search space is enhanced by good starting configurations
            treat the values in starting configuration as suggested values, then explore up and down.
        """
        # how to be guided by coarse-grained tuning
        with open(self.enhanced_starting_path, "r") as json_file:
            enhanced_starting_data = json.load(json_file)

        knob_value_range = defaultdict(list)
        for _id, data in enhanced_starting_data.items():
            config = data['config']
            if config == 'default settings':
                continue
            
            for knob, value in config.items():
                if knob not in self.dbms.knob_info.keys():
                    print(f'{knob} is not by the DBMS under specific version')
                    continue

                info = self.dbms.knob_info[knob]
                vartype = info['vartype']
                if vartype == 'string':
                    continue

                unit = info["unit"]
                if unit:
                    value = self._transfer_unit(value)
                    value = self._type_transfer(vartype, value)

                if value not in knob_value_range[knob]:
                    knob_value_range[knob].append(value)
        
        # print("knob_value_range", knob_value_range)
        # exit()

        for knob, values in knob_value_range.items():
            print(f"Defining fine search space for knob: {knob}")
            info = self.dbms.knob_info[knob]
            knob_type = info["vartype"] 
            if knob_type == "enum" or knob_type == "bool":
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
                continue
            
            boot_value = info["reset_val"]
            unit = info["unit"]
            min_value = info["min_val"]
            max_value = info["max_val"]
            print("min_value", min_value)
            print("max_value", max_value)
            suggested_values = values
            print("values", values)

            # Since the upper bound of some knob in mysql is too big, use GPT's offered upperbound for mysql
            if isinstance(self.dbms, MysqlDBMS):
                if max_value >= sys.maxsize / 10:  
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
            
            # type transformation
            try:
                suggested_values = [self._type_transfer(knob_type, value) for value in suggested_values]
                min_value = self._type_transfer(knob_type, min_value)
                max_value = self._type_transfer(knob_type, max_value)
                boot_value = self._type_transfer(knob_type, boot_value)
            except:

                def match_num(value):
                    pattern = r"(\d+)"
                    match = re.match(pattern, value)
                    if match:
                        return match.group(1)
                    else:
                        return ""

                pattern = r"(\d+)"
                suggested_values = [self._type_transfer(knob_type, re.match(pattern, value).group(1)) for value in suggested_values if re.match(pattern, value) is not None]
                min_value = self._type_transfer(knob_type, match_num(min_value))
                max_value = self._type_transfer(knob_type, match_num(max_value))
                boot_value = self._type_transfer(knob_type, match_num(boot_value))
                
            # the search space of fine-grained stage should be superset of that of coarse stage

            print("values after transferring", suggested_values)
            coarse_sequence = []
            if boot_value > sys.maxsize / 10:
                boot_value = sys.maxsize / 10

            min_value = min(min_value, boot_value)
            max_value = max(max_value, boot_value)
            # scale up and down the suggested value
            for value in suggested_values:
                for factor in self.factors:
                    # explore_up = value + factor * (max_value - value)
                    # explore_down = value + factor * (min_value - value)
                    explore_up = value + factor * (value + 1)
                    explore_down = value + factor * (min_value - value)
                    # print("explore_up", explore_up)
                    # print("explore_down", explore_down)
                    # print("sys.maxsize / 10", sys.maxsize / 10)
                    # print("explore_up < sys.maxsize / 10?", explore_up < sys.maxsize / 10)
                    if explore_up < sys.maxsize / 10 and explore_down < explore_up:
                        coarse_sequence.append(explore_up)
                        coarse_sequence.append(explore_down)
            # print("coarse_sequence after exploring up and down:", coarse_sequence)
            if max_value > sys.maxsize / 10:
                max_value = sys.maxsize / 10
            
            if min_value > sys.maxsize / 10:
                min_value = sys.maxsize / 10

            coarse_sequence = [value for value in coarse_sequence if value < sys.maxsize / 10]
            
            if knob_type == "integer":  
                coarse_sequence = [int(value) for value in coarse_sequence]
                min_value = min(min_value, min(coarse_sequence))
                max_value = max(max_value, max(coarse_sequence))
                normal_para = UniformIntegerHyperparameter(
                    knob, 
                    int(min_value), 
                    int(max_value),
                    default_value = int(boot_value),
                )
                # print(normal_para)
                self.search_space.add_hyperparameter(normal_para)
                
            elif knob_type == "real":
                coarse_sequence = [float(value) for value in coarse_sequence]
                min_value = min(min_value, min(coarse_sequence))
                max_value = max(max_value, max(coarse_sequence))
                normal_para = UniformFloatHyperparameter(
                    knob,
                    float(min_value),
                    float(max_value),
                    default_value = float(boot_value),
                )
                # print(normal_para)
                self.search_space.add_hyperparameter(normal_para)
            else:
                info = self.dbms.knob_info[knob]
                if info is None:
                    continue
                knob = self.get_default_space(knob, info)
                # print(knob)
                self.search_space.add_hyperparameter(knob)

    def define_enhancedstartings_with_enhancedspace2(self):
        """
        enhanced by feeding good starting configurations 
        but the search space is enhanced by good starting configurations
            use the knobs as pre-selected knobs
            use the min and max of values as search space
        """
        # how to be guided by coarse-grained tuning
        with open(self.enhanced_starting_path, "r") as json_file:
            enhanced_starting_data = json.load(json_file)

        knob_value_range = defaultdict(list)
        for _id, data in enhanced_starting_data.items():
            config = data['config']
            if config == 'default settings':
                continue
            
            for knob, value in config.items():
                if knob not in self.dbms.knob_info.keys():
                    print(f'{knob} is not by the DBMS under specific version')
                    continue

                info = self.dbms.knob_info[knob]
                vartype = info['vartype']
                if vartype == 'string':
                    continue

                unit = info["unit"]
                if unit:
                    value = self._transfer_unit(value)
                    value = self._type_transfer(vartype, value)

                knob_value_range[knob].append(value)

        for knob, values in knob_value_range.items():
            print(f"Defining fine search space for knob: {knob} [{values}]")
            info = self.dbms.knob_info[knob]
            knob_type = info["vartype"] 
            distinct_values = list(set(values))  # reduce redundant values
            if len(distinct_values) == 1: # the knob's values did not changes among good configurations, treat as constant
                if knob_type == "enum" or knob_type == "bool":
                    knob = self.get_default_space(knob, info)
                    self.search_space.add_hyperparameter(knob)
                    continue
                elif  knob_type == "integer":
                    const_para = Constant(knob, int(distinct_values[0]))
                    # value = int(distinct_values[0])
                    # boot_value = info["reset_val"]
                    # suggested_min = value - 0.1 * value # explore down 10%
                    # suggested_max += value + 0.1 * value # explore up 10%
                    # normal_para = UniformIntegerHyperparameter(
                    #     knob, 
                    #     int(suggested_min), 
                    #     int(suggested_max),
                    #     # default_value = int(boot_value),
                    # )
                    self.search_space.add_hyperparameter(const_para)
                else:
                    const_para = Constant(knob, float(distinct_values[0]))
                    # value = float(distinct_values[0])
                    # boot_value = info["reset_val"]
                    # suggested_min = value - 0.1 * value # explore down 10%
                    # suggested_max += value + 0.1 * value # explore up 10%
                    # normal_para = UniformFloatHyperparameter(
                    #     knob, 
                    #     float(suggested_min), 
                    #     float(suggested_max),
                    #     # default_value = float(boot_value),
                    # )
                    self.search_space.add_hyperparameter(const_para)
            else:
                if knob_type == "enum" or knob_type == "bool":
                    knob = self.get_default_space(knob, info)
                    self.search_space.add_hyperparameter(knob)
                    continue
            
                boot_value = info["reset_val"]
                unit = info["unit"]
                min_value = info["min_val"]
                max_value = info["max_val"]
                print("min_value", min_value)
                print("max_value", max_value)
                suggested_values = values
                print("values", values)

                # Since the upper bound of some knob in mysql is too big, use GPT's offered upperbound for mysql
                if isinstance(self.dbms, MysqlDBMS):
                    if max_value >= sys.maxsize / 10:  
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
                
                # type transformation
                try:
                    suggested_values = [self._type_transfer(knob_type, value) for value in suggested_values]
                    min_value = self._type_transfer(knob_type, min_value)
                    max_value = self._type_transfer(knob_type, max_value)
                    boot_value = self._type_transfer(knob_type, boot_value)
                except:

                    def match_num(value):
                        pattern = r"(\d+)"
                        match = re.match(pattern, value)
                        if match:
                            return match.group(1)
                        else:
                            return ""

                    pattern = r"(\d+)"
                    suggested_values = [self._type_transfer(knob_type, re.match(pattern, value).group(1)) for value in suggested_values if re.match(pattern, value) is not None]
                    min_value = self._type_transfer(knob_type, match_num(min_value))
                    max_value = self._type_transfer(knob_type, match_num(max_value))
                    boot_value = self._type_transfer(knob_type, match_num(boot_value))
                    
                # the search space of fine-grained stage should be superset of that of coarse stage
                print("values after transferring", suggested_values)

                suggested_min = min(suggested_values)
                suggested_min -= 0.1 * suggested_min # explore down 10%
                suggested_max = max(suggested_values)
                suggested_max += 0.1 * suggested_max # explore up 10%

                if boot_value > sys.maxsize / 10:
                    boot_value = sys.maxsize / 10

                min_value = min(min_value, boot_value)
                max_value = max(max_value, boot_value)

                # print("coarse_sequence after exploring up and down:", coarse_sequence)
                if max_value > sys.maxsize / 10:
                    max_value = sys.maxsize / 10
                
                if min_value > sys.maxsize / 10:
                    min_value = sys.maxsize / 10
                
                if knob_type == "integer":  
                    min_value = min(min_value, suggested_min)
                    max_value = max(max_value, suggested_max)
                    normal_para = UniformIntegerHyperparameter(
                        knob, 
                        int(min_value), 
                        int(max_value),
                        default_value = int(boot_value),
                    )
                    # print(normal_para)
                    self.search_space.add_hyperparameter(normal_para)
                    
                elif knob_type == "real":
                    min_value = min(min_value, suggested_min)
                    max_value = max(max_value, suggested_max)
                    normal_para = UniformFloatHyperparameter(
                        knob,
                        float(min_value),
                        float(max_value),
                        default_value = float(boot_value),
                    )
                    # print(normal_para)
                    self.search_space.add_hyperparameter(normal_para)
                else:
                    info = self.dbms.knob_info[knob]
                    if info is None:
                        continue
                    knob = self.get_default_space(knob, info)
                    # print(knob)
                    self.search_space.add_hyperparameter(knob)

    