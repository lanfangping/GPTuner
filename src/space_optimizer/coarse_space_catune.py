from space_optimizer.default_space import DefaultSpace
from dbms.mysql import MysqlDBMS
import sys
import os
import json
import re
from smac import HyperparameterOptimizationFacade, Scenario, initial_design
from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    EqualsCondition,
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,

)

from search_space.knowledge_based_space import attach_sampler_to_configspace, unify_unit

class CoarseSpaceCATune(DefaultSpace):

    def __init__(self, dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log):
        coarse_results_path = os.path.join(results_folder, f"{dbms.name}/coarse_rules")
        super().__init__(dbms, test, timeout, target_knobs_path, coarse_results_path, seed, log)
        
        self.special_skill_path = special_skill_path
        self.log = log
        self.get_special_info()
        self.define_search_space()
        self.log.info(f"Coarse Configuration Space: {self.search_space}")

    def define_search_space(self):
        self.log.info("CoarseSpaceCATune - Building coarse search space...")
        for knob in self.target_knobs:
            print(knob)
            info = self.dbms.knob_info[knob]
            if info is None:
                self.target_knobs.remove(knob) # this knob is not by the DBMS under specific version
                continue

            suggest_info = self.suggest_knob_info.get(knob, None)
            if suggest_info is None:
                info = self.dbms.knob_info[knob]
                if info is None:
                    return []
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
                return []
            
            boot_value = info["reset_val"]
            knob_type = info["vartype"]
            if knob_type == "enum" or knob_type == "bool":
                knob = self.get_default_space(knob, info)
                self.search_space.add_hyperparameter(knob)
                continue

            sequence = self.get_sequence_from_coarse(knob=knob)

            if knob_type == "integer":
                normal_para = CategoricalHyperparameter(
                    knob,
                    [int(value) for value in sequence],
                    default_value = int(boot_value),
                )
            else:
                normal_para = CategoricalHyperparameter(
                    knob,
                    [float(value) for value in sequence],
                    default_value = float(boot_value),
                )
            
            if self.suggest_knob_info[knob]['is_special']:
                control_para = CategoricalHyperparameter(f"control_{knob}", ["0", "1"], default_value="0") 
                if type(self.suggest_knob_info[knob]['special_value']) is list:
                    # special_para = OrdinalHyperparameter(f"special_{knob}", [int(value) for value in special_value])
                    special_para = CategoricalHyperparameter(f"special_{knob}", [eval(value) for value in self.suggest_knob_info[knob]['special_value']])
                else:
                    special_para = Constant(f"special_{knob}", int(self.suggest_knob_info[knob]['special_value']))

                self.search_space.add_hyperparameters([control_para, normal_para, special_para])
                
                normal_cond = EqualsCondition(self.search_space[knob], self.search_space[f"control_{knob}"], "0")
                special_cond = EqualsCondition(self.search_space[f"special_{knob}"], self.search_space[f"control_{knob}"], "1")
                
                self.search_space.add_conditions([normal_cond, special_cond])
            else:
                self.search_space.add_hyperparameter(normal_para)

    def get_default_space(self, knob_name, info):
        boot_value = info["reset_val"]
        min_value = info["min_val"]
        max_value = info["max_val"]
        self.log.info(f"Default search space for knob {knob_name}: min={min_value}, max={max_value}, boot={boot_value}")
        boot_value = unify_unit(boot_value, info["unit"])
        min_value = unify_unit(min_value, info["unit"])
        max_value = unify_unit(max_value, info["unit"])
        self.log.info(f"Unified default search space for knob {knob_name}: min={min_value}, max={max_value}, boot={boot_value}")
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
                [enum_val for enum_val in info["enumvals"]],
                default_value = str(boot_value),
            )
        elif knob_type == "bool":
            knob = CategoricalHyperparameter(
                knob_name,
                ["on", "off"],
                default_value = str(boot_value)
            )
        return knob