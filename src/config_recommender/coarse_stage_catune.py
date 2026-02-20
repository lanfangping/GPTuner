from space_optimizer.coarse_space_catune import CoarseSpaceCATune
from smac import HyperparameterOptimizationFacade, Scenario, initial_design
import os, json, sys, re
# Package from CATune
from optimizer.topo_latin_hypercube_design import TopoLatinHypercubeInitialDesign
from run_SMAC import setup_rules
from search_space.knowledge_based_space import attach_sampler_to_configspace, unify_unit
from sampler.topological_sampler import TopoSampler

from ConfigSpace import (
    CategoricalHyperparameter,
    Constant,
    EqualsCondition,
    ConfigurationSpace,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,

)

from dbms.mysql import MysqlDBMS

class CoarseStageCATune(CoarseSpaceCATune):

    def __init__(self, dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log, rules=[], conditional_activations=[]):
        super().__init__(dbms, test, timeout, target_knobs_path, special_skill_path, results_folder, seed, log)
        self.rules = rules
        self.conditional_activations = conditional_activations
        knob_info_path = os.path.join(os.path.dirname(os.path.dirname(special_skill_path)), 'knob_info/system_view.json')
        self.all_knob_info = {}
        for knob, knob_info in json.load(open(knob_info_path, "r")).items():
            knob_info['default'] = knob_info['boot_val']
            knob_info['type'] = knob_info['vartype']
            self.all_knob_info[knob] = knob_info

    

    def optimize(self, name, trials_number, initial_config_number, strategy='adaptive'):
        scenario = Scenario(
            configspace=self.search_space,
            name=name,
            seed=self.seed,
            deterministic=True,
            n_trials=trials_number,
            use_default_config=True,
        )

        if strategy is not None:
            topo_sampler = TopoSampler(
                    cs=self.search_space, 
                    constraints=self.rules, 
                    conditional_activations=self.conditional_activations, 
                    seed=self.seed,
                    strategy=strategy
                )
            self.search_space = setup_rules(self.search_space, self.rules, self.all_knob_info)
            attach_sampler_to_configspace(cs=self.search_space, sampler=topo_sampler, method_name='sample_configuration')
            init_design = TopoLatinHypercubeInitialDesign(
                scenario=scenario,
                topo_sampler=topo_sampler,
                n_configs=initial_config_number,
                max_ratio=0.8,
                seed=self.seed,
            )
        else:
            init_design = initial_design.LatinHypercubeInitialDesign(
                scenario,
                n_configs=initial_config_number,
                max_ratio=0.8,  # set this to a value close to 1 to get exact initial_configs as specified
                seed=self.seed
            )
        
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            initial_design=init_design,
            target_function=self.set_and_replay,
            overwrite=False,
        )
        
        smac.optimize()