from abc import ABC, abstractmethod
from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
from dbms.postgres import PgDBMS
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

from search_space.knowledge_based_space import unify_unit

class FineSpaceCATune(DefaultSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log):
        fine_results_path = os.path.join(results_folder, f"{dbms.name}/fine_rules")
        super().__init__(dbms, test, timeout, target_knobs_path, fine_results_path, seed, log)
        self.factors = [0, 0.25, 0.5]
        self.special_skill_path = special_skill_path
        self.conditional_activations = []
        self.get_special_info()
        self.define_search_space()
        print(f"conditional_activations: {self.conditional_activations}")
        log.info(f"Fine Configuration Space: {self.search_space}")
        self.coarse_path = os.path.join(results_folder, f"{self.dbms.name}/coarse_rules/{self.seed}/runhistory.json") 

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
            
            
            suggest_info = self.suggest_knob_info.get(knob, None)
            if suggest_info is None:
                info = self.dbms.knob_info[knob]
                if info is None:
                    continue
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
                continue

            # print(f"Defining fine search space for knob: {knob}")
            boot_value = info["reset_val"]
            unit = info["unit"]
            knob_type = info["vartype"]

            boot_value = info["reset_val"]
            knob_type = info["vartype"]
            coarse_sequence = self.get_sequence_from_coarse(knob=knob)

            min_value = suggest_info["min_value"]
            if min_value is None:
                min_value = info["min_val"]
            
            max_value = suggest_info["max_value"]
            if max_value is None:
                max_value = info["max_val"]

            # unify the number based on the unit, then convert the data type(int, float)
            min_value = self._type_transfer(knob_type, unify_unit(min_value, suggest_info['min_unit']))
            max_value = self._type_transfer(knob_type, unify_unit(max_value, suggest_info['max_unit']))
            boot_value = self._type_transfer(knob_type, unify_unit(boot_value, unit))

            # if knob == 'seq_page_cost':
            #     print(f"Min value: {min_value}, max_value: {max_value}, boot_value: {boot_value} for knob {knob}")
            
            if boot_value > sys.maxsize / 10:
                boot_value = sys.maxsize / 10
            
            if max_value > sys.maxsize / 10:
                max_value = sys.maxsize / 10

            min_value = min(min_value, boot_value)
            max_value = max(max_value, boot_value)
            # if knob == 'seq_page_cost':
            #     print(f"Min value: {min_value}, max_value: {max_value}, boot_value: {boot_value} for knob {knob}")
            #     print(f"Coarse sequence for knob {knob}: {coarse_sequence}")
            
            # print(f"Coarse sequence for knob {knob}: {coarse_sequence}")
            # print(f"Min value: {min_value}, max_value: {max_value} for knob {knob}")
            min_value = min(min_value, min(coarse_sequence))
            max_value = max(max_value, max(coarse_sequence))
            # if knob == 'seq_page_cost':
            #     print(f"Min value: {min_value}, max_value: {max_value}, boot_value: {boot_value} for knob {knob}")
            # print(f"Final min_value: {min_value}, max_value: {max_value} for knob {knob}")
            if min_value == max_value: # the parameter has the equal min_value and max_value
                normal_para = Constant(knob, int(max_value))
            elif min_value > max_value:
                print(f"error: min_value is lager than max_value: {min_value} > {max_value}, skip")
                continue
            else:
                if knob_type == "integer":
                    normal_para = UniformIntegerHyperparameter(
                        knob, 
                        int(min_value), 
                        int(max_value),
                        default_value = int(boot_value),
                    )
                else: # real
                    normal_para = UniformFloatHyperparameter(
                        knob,
                        float(min_value),
                        float(max_value),
                        default_value = float(boot_value),
                    )
                
                # if knob == 'seq_page_cost':
                #     print(f"Defined normal_para for knob {knob}: {normal_para}")
                #     exit()
                if self.suggest_knob_info[knob]["is_special"]:
                    # 0 normal, 1 special
                    control_para = CategoricalHyperparameter(f"control_{knob}", ["0", "1"], default_value="0") 
                    if type(self.suggest_knob_info[knob]["special_value"]) is list:
                        special_para = CategoricalHyperparameter(f"special_{knob}", [eval(str(value)) for value in self.suggest_knob_info[knob]["special_value"]])
                    else:
                        special_para = Constant(f"special_{knob}", self.suggest_knob_info[knob]["special_value"])

                    self.search_space.add_hyperparameters([control_para, normal_para, special_para])
                    normal_cond = EqualsCondition(self.search_space[knob], self.search_space[f"control_{knob}"], "0")
                    special_cond = EqualsCondition(self.search_space[f"special_{knob}"], self.search_space[f"control_{knob}"], "1")
                    self.conditional_activations.append((f"{knob}", f"control_{knob}", {"0"}))
                    self.conditional_activations.append((f"special_{knob}", f"control_{knob}", {"1"}))
                    self.search_space.add_conditions([normal_cond, special_cond])
                else:
                    self.search_space.add_hyperparameter(normal_para)

    
    def get_default_space(self, knob_name, info):
        boot_value = info["reset_val"]
        min_value = info["min_val"]
        max_value = info["max_val"]
        knob_type = info["vartype"]
        self.error_log.info(f"Boot value: {boot_value}, min value: {min_value}, max value: {max_value} for knob {knob_name}")
        boot_value = unify_unit(boot_value, info["unit"])
        min_value = unify_unit(min_value, info["unit"])
        max_value = unify_unit(max_value, info["unit"])
        self.error_log.info(f"Unified boot value: {boot_value}, unified min value: {min_value}, unified max value: {max_value} for knob {knob_name}")

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


    