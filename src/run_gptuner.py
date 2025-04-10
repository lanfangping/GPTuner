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
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 
        os.makedirs(os.path.join(f"{folder_path}", "temp_results"))
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knob_info"))
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources/gpt"))
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources/web"))
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources/manual"))
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/normal"))
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/special"))
        os.makedirs(os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/tuning_lake"))
        os.makedirs(os.path.join(f"{folder_path}", f"{args.db}")) # optimization results
        os.makedirs(os.path.join(f"{folder_path}", f"{args.db}", "log"))
    
    source_web_folder = "knowledge_collection/postgres/knowledge_sources/web"
    dest_web_folder = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knowledge_sources/web")
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
    parser.add_argument("--db", type=str)
    parser.add_argument("--database", type=str)
    parser.add_argument("--test", type=str)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--result_path", type=str, default="./experiments_results/test")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--restart_cmd", type=str, default="sudo restart tpcc_workload")
    parser.add_argument("--recover_script", type=str, default="./scripts/recover_docker_postgres.sh")
    args = parser.parse_args()
    misc.over_write_args_from_file(args, args.config)

    # store the optimization results
    # Derive output folder from script name
    script_name = args.config.split('.')[0].split('/')[-1]
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    # folder_name = f'{script_name}_{current_time}'
    folder_name = "test_config_202504072324"
    folder_path = f"./experiments_results/{folder_name}"
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
    else:
        api_base = os.environ.get("DEEPSEEK_API_BASE")
        api_key = os.environ.get("DEEPSEEK_API_KEY")

    # Select target knobs, write your api_base and api_key
    dbms._connect(args.database)
    knob_selection = KnobSelection(db=args.db, dbms=dbms, benchmark=args.test, knowledge_path=folder_path, api_base=api_base, api_key=api_key, model=args.model)
    knob_selection.select_interdependent_all_knobs()

    # prepare tuning lake and structured knowledge
    # f"/home/knob/revision/GPTuner/knowledge_collection/{args.db}/target_knobs.txt"
    target_knobs_path = os.path.join(folder_path, "knowledge_collection", f"{args.db}", "target_knobs.txt") 
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
                
    
    # knowledge_pre = KGPre(db=args.db, api_base=api_base, api_key=api_key, model=args.model, knowledge_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}"))
    # knowledge_trans = KGTrans(db=args.db, api_base=api_base, api_key=api_key, model=args.model, knowledge_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}"))
    # knowledge_update = KGUpdate(db=args.db, api_base=api_base, api_key=api_key, model=args.model, knowledge_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}"))
    # for i in range(1, 6):
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #         futures = {executor.submit(process_knob, knob, knowledge_pre, knowledge_trans, knowledge_update): knob for knob in target_knobs}
    #         for future in concurrent.futures.as_completed(futures):
    #             print(future.result())
    #     print(f"Update {i} completed")

    # if args.db == 'postgres':
    #     # config_path = "./configs/postgres.ini"
    #     # config.read(config_path)
    #     # dbms = PgDBMS.from_file(config)
    #     dbms = PgDBMS(db=args.database, user=args.user, password=args.password, restart_cmd=args.restart_cmd, recover_script=args.recover_script, knob_info_path=os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/knob_info/system_view.json"))
    # elif args.db == 'mysql':
    #     config_path = "./configs/mysql.ini"
    #     config.read(config_path)
    #     dbms = MysqlDBMS.from_file(config)
    # else:
    #     raise ValueError("Illegal dbms!")
    
    # store the optimization results
    # folder_path = "../optimization_results/"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)  

    special_skill_path = os.path.join(f"{folder_path}", f"knowledge_collection/{args.db}/structured_knowledge/special")
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

    
    # gptuner_fine = FineStage(
    #     dbms=dbms, 
    #     target_knobs_path=target_knobs_path, 
    #     test=args.test, 
    #     timeout=args.timeout, 
    #     seed=args.seed,
    #     special_skill_path=special_skill_path,
    #     log=log,
    #     results_folder = folder_path 
    # )

    # gptuner_fine.optimize(
    #     name = os.path.join(f".{folder_path}", f"{args.db}", "fine"), # f"../optimization_results/{args.db}/fine/", 
    #     trials_number=110 # history trials + new tirals
    # )   

