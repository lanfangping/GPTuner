from configparser import ConfigParser
import argparse
import time
import os
import openai
import concurrent.futures
from knowledge_handler.knowledge_update import KGUpdate
# from dbms.postgres import PgDBMS
from dbms.postgres_docker import PgDBMS
from dbms.mysql import  MysqlDBMS
from config_recommender.coarse_stage import CoarseStage
from config_recommender.fine_stage import FineStage
from knowledge_handler.knowledge_preparation import KGPre
from knowledge_handler.knowledge_transformation import KGTrans
from space_optimizer.knob_selection import KnobSelection
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
from utils import logger, misc
from datetime import datetime
# import logging

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

# # Derive output folder from script name
# script_name = os.path.splitext(os.path.basename(__file__))[0]
# output_folder = f"output/{script_name}"
# os.makedirs(output_folder, exist_ok=True)

# # Configure logging
# log_file = os.path.join(output_folder, "log.txt")
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler(),  # Print to console
#         logging.FileHandler(log_file)  # Write to log file
#     ]
# )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default='postgres')
    parser.add_argument("--test", type=str, default='tpcc') # workload type
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--enhanced", action='store_true')
    parser.add_argument("--coarse", type=str, default='knowledge')
    parser.add_argument("--fine", type=str, default='knowledge')
    parser.add_argument("--mode", type=str, default='past_best')
    parser.add_argument("--enhanced_starting_path", type=str, default='../DBtuningDataset/historical_best_config/historical_best_tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202412032307.json')
    parser.add_argument("--enhanced_strategy", type=str, default='suggest_values')
    parser.add_argument("--config", type=str, default='') # config file
    args = parser.parse_args()
    misc.over_write_args_from_file(args, args.config)

    # store the optimization results
    # Derive output folder from script name
    script_name = args.config.split('.')[0].split('/')[-1]
    # folder_name="optimization_results_enhancedstarting_wellsearchspace"
    current_time = datetime.now().strftime("%Y%m%d%H%M")
    folder_name = f'{script_name}_{current_time}'
    folder_path = f"./experiments_results/{folder_name}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 
        os.makedirs(f"{folder_path}temp_results/")
        os.makedirs(f"{folder_path}{args.db}/")  
        os.makedirs(f"{folder_path}{args.db}/log/") 

    logger_level = "INFO"
    log = logger.get_logger(script_name, folder_path, logger_level)
    
    log.info(f'Input arguments: {args}')
    log.info(f'Results stored path: {folder_path}')
    time.sleep(2)

    config = ConfigParser()
    
    if args.db == 'postgres':
        config_path = "./configs/postgres.ini"
        config.read(config_path)
        dbms = PgDBMS.from_file(config)
    elif args.db == 'mysql':
        config_path = "./configs/mysql.ini"
        config.read(config_path)
        dbms = MysqlDBMS.from_file(config)
    else:
        raise ValueError("Illegal dbms!")


    # Select target knobs, write your api_base and api_key
    dbms._connect("benchbase")
    api_key=os.environ.get("DEEPSEEK_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE")
    model = "deepseek-chat"
    # knob_selection = KnobSelection(db=args.db, dbms=dbms, benchmark=args.test, api_base=api_base, api_key=api_key, model=model)
    # knob_selection.select_interdependent_all_knobs()
    dbms._disconnect()

    # prepare tuning lake and structured knowledge
    target_knobs_path = f"./knowledge_collection/{args.db}/target_knobs.txt"
    with open(target_knobs_path, 'r') as file:
        lines = file.readlines()
        target_knobs = [line.strip() for line in lines]


    # # write your api_base and api_key
    # knowledge_pre = KGPre(db=args.db, api_base=api_base, api_key=api_key, model=model)
    # knowledge_trans = KGTrans(db=args.db, api_base=api_base, api_key=api_key, model=model)
    # knowledge_update = KGUpdate(db=args.db, api_base=api_base, api_key=api_key, model=model)

    # for i, knob in enumerate(target_knobs):
    #     print(f"{i}th, total {len(target_knobs)} knobs")
    #     try: 
    #         process_knob(knob, knowledge_pre, knowledge_trans, knowledge_update)
    #     except KeyError as e:
    #         print(f"{e}")
    #         continue

    # for i in range(1, 6):
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #         futures = {executor.submit(process_knob, knob, knowledge_pre, knowledge_trans, knowledge_update): knob for knob in target_knobs}
    #         for future in concurrent.futures.as_completed(futures):
    #             print(future.result())
    #     print(f"Update {i} completed")
    #     print("===============================\n")


    if args.db == 'postgres':
        config_path = "./configs/postgres.ini"
        config.read(config_path)
        dbms = PgDBMS.from_file(config)
    elif args.db == 'mysql':
        config_path = "./configs/mysql.ini"
        config.read(config_path)
        dbms = MysqlDBMS.from_file(config)
    else:
        raise ValueError("Illegal dbms!")
    
     

    if not args.enhanced:
        gptuner_coarse = CoarseStage(
            dbms=dbms, 
            target_knobs_path=target_knobs_path, 
            test=args.test,  # workload type
            timeout=args.timeout, 
            seed=args.seed,
            enhanced=args.enhanced,
            enhanced_starting_path=args.enhanced_starting_path,
            coarse=args.coarse,
            log=log,
            folder_name=folder_name
        )
        
        gptuner_coarse.optimize(
            name = f"../experiments_results/{folder_name}/{args.db}/coarse/", 
            trials_number=30, 
            initial_config_number=10)
        time.sleep(20)

    gptuner_fine = FineStage(
        dbms=dbms, 
        target_knobs_path=target_knobs_path, 
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed,
        enhanced=args.enhanced,
        enhanced_starting_path=args.enhanced_starting_path,
        enhanced_strategy=args.enhanced_strategy,
        fine=args.fine,
        coarse_folder_name=folder_name,
        log=log
    )

    if not args.enhanced:
        gptuner_fine.optimize(
            name = f"../experiments_results/{folder_name}/{args.db}/fine/",
            trials_number=110 # history trials + new tirals
        )   
    else:
        if args.fine.lower() == 'knowledge':
            gptuner_fine.optimize_enhanced_starting_wellsearchspace(
                name = f"../experiments_results/{folder_name}/{args.db}/fine/",
                trials_number=110 # history trials + new tirals
            )
        elif args.fine.lower() == 'default':
            gptuner_fine.optimize_enhanced_starting(
                name = f"../experiments_results/{folder_name}/{args.db}/fine/",
                trials_number=110 # history trials + new tirals
            )


