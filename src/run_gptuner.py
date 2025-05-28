from configparser import ConfigParser
import argparse
import time
from datetime import datetime
import os
import json
import openai
import concurrent.futures
import subprocess
from knowledge_handler.knowledge_update import KGUpdate
from knowledge_handler.utils import get_hardware_info, get_disk_type
from dbms.postgres import PgDBMS
from dbms.mysql import  MysqlDBMS
from config_recommender.coarse_stage import CoarseStage
from config_recommender.fine_stage import FineStage
from knowledge_handler.knowledge_preparation import KGPre
from knowledge_handler.knowledge_transformation import KGTrans
from space_optimizer.knob_selection import KnobSelection
from utils import misc
from utils.logger import MyLogger
from dotenv import load_dotenv
from utils.exp_tools import replace_range_for_knobs, replace_special_values
load_dotenv()  # take environment variables from .env.

def process_knob(knob, knowledge_pre, knowledge_trans, knowledge_update):
    try:
        knowledge_pre.pipeline(knob)
        knowledge_trans.pipeline(knob)
        new_structure = knowledge_update.pipeline(knob)
        if new_structure is False:
            return f"Skipped processing for {knob}"
        return f"Processed {knob}"
    except openai.RateLimitError as e:
        wait_time = float(e.response.headers.get('Retry-After', 0.5))
        print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
        time.sleep(wait_time)
        return process_knob(knob, knowledge_pre, knowledge_trans, knowledge_update)  # Retry recursively after waiting

def make_folders(folder_path, args):
    """
    folder_path: the folder stores the optimization details
    """
    # if not os.path.exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True) 
        os.makedirs(os.path.join(f"{folder_path}", "temp_results"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knob_info"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources/gpt"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources/web"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources/manual"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/normal"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/special"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/tuning_lake"), exist_ok=True)
        os.makedirs(os.path.join(f"{folder_path}", f"{args.db}"), exist_ok=True) # optimization results
        os.makedirs(os.path.join(f"{folder_path}", f"{args.db}", "log"), exist_ok=True)
    except Exception as e:
        print(f"Warning: {e}")
    
    source_web_folder = "knowledge_collection/postgres/knowledge_sources/web"
    dest_web_folder = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources")
    source_official_document = "src/utils/official_document.json"
    source_system_view = "knowledge_collection/postgres/knob_info/system_view.json"
    dest_knob_info = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knob_info")
    source_candicate_knobs = "knowledge_collection/postgres/candidate_knobs.txt"
    dest_candicate_knobs = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}")
    commands = [
        f"cp -r {source_web_folder} {dest_web_folder}",
        f"cp {source_official_document} {dest_knob_info}",
        f"cp {source_system_view} {dest_knob_info}",
        f"cp {source_candicate_knobs} {dest_candicate_knobs}"
    ]
    for command in commands:
        subprocess.run(command, shell=True, check=True)
    time.sleep(2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default='') # config file
    parser.add_argument("--folder", type=str, default='default')
    parser.add_argument("--process", type=str, default='whole') # mode: whole, knowledge, optimization
    parser.add_argument("--db", type=str)
    parser.add_argument("--database", type=str)
    parser.add_argument("--test", type=str)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--result_path", type=str, default="./experiments_results/test")
    parser.add_argument("--knobs", type=str, default="None")
    parser.add_argument("--suggest_range_path", type=str, default="None")
    parser.add_argument("--suggest_range_target_path", type=str, default="None")
    parser.add_argument("--suggest_range_mode", type=str, default="default") # `default`: only use `suggest_range_path`, `narrow`: replace the range in `suggest_range_path` with the narrow range in `suggest_range_target_path`
    parser.add_argument("--suggest_values_path", type=str, default="None")
    parser.add_argument("--special_skill_path", type=str, default="None")
    parser.add_argument("--special_skill_mode", type=str, default="default") 
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--restart_cmd", type=str, default="sudo restart tpcc_workload")
    parser.add_argument("--recover_script", type=str, default="./scripts/recover_docker_postgres.sh")
    args = parser.parse_args()
    misc.over_write_args_from_file(args, args.config)

    # store the optimization results
    # Derive output folder from script name
    script_name = ".".join(args.config.split('.')[:-1]).split('/')[-1]
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    if args.folder == 'default':
        folder_name = f'{script_name}_{current_time}'
    else:
        folder_name = args.folder # "deepseek-v3-overall_202504101721"
    folder_path = f"./experiments_results/{args.test}/{folder_name}"

    setattr(args, 'result_path', folder_path)
    make_folders(folder_path=folder_path, args=args)

    logger_level = "INFO"
    log = MyLogger(script_name, folder_path, logger_level).logger
    log.info(f'Input arguments: {args}')

    config = ConfigParser()
    if args.db == 'postgres':
        # config_path = "./configs/postgres.ini"
        # config.read(config_path)
        # dbms = PgDBMS.from_file(config)
        dbms = PgDBMS(
            db=args.database, 
            user=args.user, 
            password=args.password, 
            restart_cmd=args.restart_cmd, 
            recover_script=args.recover_script, 
            knob_info_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knob_info/system_view.json"), 
            log_path=folder_path)
    elif args.db == 'mysql':
        config_path = "./configs/mysql.ini"
        config.read(config_path)
        dbms = MysqlDBMS.from_file(config)
    else:
        raise ValueError("Illegal dbms!")

    # write your api_base and api_key
    if args.model.startswith('gpt'):
        api_base = os.environ.get("OPENAI_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith('deepseek'):
        api_base = os.environ.get("DEEPSEEK_API_BASE")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    else:
        api_base = os.environ.get("LLAMA_API_BASE")
        api_key = os.environ.get("LLAMA_API_KEY")

    # f"/home/knob/revision/GPTuner/knowledge_collection/{args.db}/target_knobs.txt"
    if args.knobs != "None": # provide selected knobs
        source_selected_knobs = args.knobs
        dest_selected_knobs = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}")
        command = f"cp {source_selected_knobs} {dest_selected_knobs}"
        subprocess.run(command, shell=True, check=True)
        time.sleep(2)
    target_knobs_path = os.path.join(folder_path, "knowledge_collection", f"{args.db}", "target_knobs.txt")
    
    # prepare tuning lake and structured knowledge
    target_knobs = []
    knob_info = json.load(open(os.path.join(folder_path, f"knowledge_collection/{args.db}/knob_info/system_view.json")))
    with open(target_knobs_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            knob = line.strip()
            if knob in knob_info.keys():
                target_knobs.append(knob)
            else:
                log.warning(f"'{knob}' is not in {args.db}")

    if args.process == 'whole' or args.process == 'knowledge':
        # write your api_base and api_key
        if args.model.startswith('gpt'):
            api_base = os.environ.get("OPENAI_API_BASE")
            api_key = os.environ.get("OPENAI_API_KEY")
        elif args.model.startswith('deepseek'):
            api_base = os.environ.get("DEEPSEEK_API_BASE")
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            api_base = os.environ.get("LLAMA_API_BASE")
            api_key = os.environ.get("LLAMA_API_KEY")

        # Select target knobs, write your api_base and api_key
        dbms._connect(args.database)
        knob_selection = KnobSelection(db=args.db, dbms=dbms, benchmark=args.test, knowledge_path=folder_path, api_base=api_base, api_key=api_key, model=args.model)
        knob_selection.select_interdependent_all_knobs() # if target_knob.txt exits, then this step is skipped

        knowledge_pre = KGPre(db=args.db, api_base=api_base, api_key=api_key, model=args.model, knowledge_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}"))
        knowledge_trans = KGTrans(db=args.db, api_base=api_base, api_key=api_key, model=args.model, knowledge_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}"))
        knowledge_update = KGUpdate(db=args.db, api_base=api_base, api_key=api_key, model=args.model, knowledge_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}"))
        for i, knob in enumerate(target_knobs):
            print(f"{i}th, total {len(target_knobs)} knobs")
            try: 
                process_knob(knob, knowledge_pre, knowledge_trans, knowledge_update)
            except KeyError as e:
                log.error(f"Error for process knob '{knob}': {e}")
                continue
        # for i in range(1, 6):
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #         futures = {executor.submit(process_knob, knob, knowledge_pre, knowledge_trans, knowledge_update): knob for knob in target_knobs}
        #         for future in concurrent.futures.as_completed(futures):
        #             print(future.result())
        #     print(f"Update {i} completed")
    # if args.process == 'optimization' and args.folder == 'default':
    #     print('Please specify the knowledge folder using --folder.')
    #     exit()
    
    if args.suggest_range_path != "None" or args.suggest_values_path != "None" or args.special_skill_path != "None":
        if args.suggest_range_path == "None" or args.suggest_values_path == "None" or args.special_skill_path == "None":
            print('Please suggest_range_path, suggest_values_path, or special_skill_path cannot be None when one of them is specified.')
            exit()

        if args.special_skill_path == '.':
            source_special_skill_path = f"knowledge_collection/{args.db}/manual_collected_special"
        else:
            source_special_skill_path = os.path.join(f"{args.special_skill_path}", f"knowledge_collection/{args.db}/structured_knowledge/special")
        if args.special_skill_base_path != 'None':
            source_special_skill_base_path = os.path.join(f"{args.special_skill_base_path}", f"knowledge_collection/{args.db}/structured_knowledge/special")
        source_normal_suggest_value_skill_path = os.path.join(f"{args.suggest_values_path}", f"knowledge_collection/{args.db}/structured_knowledge/normal")
        source_normal_suggest_range_skill_path = os.path.join(f"{args.suggest_range_path}", f"knowledge_collection/{args.db}/structured_knowledge/normal")
        target_special_skill_path = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge")
        target_normal_skill_path = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/normal")
        
        if args.special_skill_mode == 'default':
            command = f"cp -r {source_special_skill_path} {target_special_skill_path}"
            subprocess.run(command, shell=True, check=True)
            time.sleep(2)
        else:
            replace_special_values(target_knobs, source_special_skill_path, source_special_skill_base_path, target_special_skill_path)
        cpu_cores, ram_size, disk_size = get_hardware_info()
        disk_type = get_disk_type()

        if args.suggest_range_mode == 'narrow':
            range_info = replace_range_for_knobs(target_knobs, args.suggest_range_path, args.suggest_range_target_path)

        with open(target_knobs_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                data = {}
                knob = line.strip()
                if knob not in dbms.knob_info.keys(): # knob is not in system_view
                    continue
                value_data = json.load(open(os.path.join(source_normal_suggest_value_skill_path, f"{knob}.json"), "r"))
                range_data = json.load(open(os.path.join(source_normal_suggest_range_skill_path, f"{knob}.json"), "r"))
                if args.suggest_range_mode != 'default':
                    data['min_value'] = range_info[knob]['min_value']
                    data['max_value'] = range_info[knob]['max_value']
                else:
                    data['min_value'] = range_data['min_value']
                    data['max_value'] = range_data['max_value']
                data['suggested_values'] = value_data['suggested_values']
                data.update({"cpu":cpu_cores, "ram":ram_size, "disk_size":disk_size, "disk_type":disk_type})
                json.dump(data, open(os.path.join(target_normal_skill_path, f"{knob}.json"), "w"))
    elif args.process == 'optimization' and args.folder == 'default':
        print('Please specify the knowledge folder using --folder.')
        exit()

    exit()
    if args.process == 'whole' or args.process == 'optimization':
        special_skill_path = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/special")
        normal_skill_path = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/normal")
        gptuner_coarse = CoarseStage(
            dbms=dbms, 
            target_knobs_path=target_knobs_path, 
            test=args.test, 
            timeout=args.timeout, 
            seed=args.seed,
            special_skill_path=special_skill_path,
            log=log,
            results_folder = folder_path
        )

        gptuner_coarse.optimize(
            name = os.path.join(f".{folder_path}", f"{args.db}", "coarse"),  # f"../optimization_results/{args.db}/coarse/", 
            trials_number=30, 
            initial_config_number=10
            )
        time.sleep(2)

        
        gptuner_fine = FineStage(
            dbms=dbms, 
            target_knobs_path=target_knobs_path, 
            test=args.test, 
            timeout=args.timeout, 
            seed=args.seed,
            special_skill_path=special_skill_path,
            log=log,
            results_folder = folder_path 
        )

        gptuner_fine.optimize(
            name = os.path.join(f".{folder_path}", f"{args.db}", "fine"), # f"../optimization_results/{args.db}/fine/", 
            trials_number=110 # history trials + new tirals
        )   

