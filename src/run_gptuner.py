from configparser import ConfigParser
import argparse
import time
import os
import openai
import concurrent.futures
from datetime import datetime
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("test", type=str) # workload type
    parser.add_argument("timeout", type=int)
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-kw", type=int, default=1)
    args = parser.parse_args()
    print(f'Input arguments: {args}')
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

    target_knobs_path = f"./knowledge_collection/{args.db}/target_knobs.txt"
    # Select target knobs, write your api_base and api_key
    if args.kw == 1 or args.kw == '1':
        dbms._connect("benchbase")
        api_key=os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE")
        model = os.environ.get("OPENAI_MODEL")
        knob_selection = KnobSelection(db=args.db, dbms=dbms, benchmark=args.test, api_base=api_base, api_key=api_key, model=model)
        knob_selection.select_interdependent_all_knobs()
        dbms._disconnect()

        # prepare tuning lake and structured knowledge
        with open(target_knobs_path, 'r') as file:
            lines = file.readlines()
            target_knobs = [line.strip() for line in lines]

        # write your api_base and api_key
        knowledge_pre = KGPre(db=args.db, api_base=api_base, api_key=api_key, model=model)
        knowledge_trans = KGTrans(db=args.db, api_base=api_base, api_key=api_key, model=model)
        knowledge_update = KGUpdate(db=args.db, api_base=api_base, api_key=api_key, model=model)

        for i, knob in enumerate(target_knobs):
            print(f"{i}th, total {len(target_knobs)} knobs")
            try: 
                process_knob(knob, knowledge_pre, knowledge_trans, knowledge_update)
            except KeyError as e:
                print(f"{e}")
                continue

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
    
    # store the optimization results
    # current_time = datetime.now().strftime("%Y%m%d%H%M")
    current_time = 202503231209
    folder = f"optimization_results/run_{current_time}"
    folder_path = f"./{folder}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  
        os.makedirs(os.path.join(folder_path, 'temp_results'))
        os.makedirs(os.path.join(folder_path, f'{args.db}', 'log'))

    gptuner_coarse = CoarseStage(
        dbms=dbms, 
        target_knobs_path=target_knobs_path, 
        test=args.test,  # workload type
        timeout=args.timeout, 
        seed=args.seed,
        folder=folder   
    )

    gptuner_coarse.optimize(
        name = f"../{folder}/{args.db}/coarse/", 
        trials_number=30, 
        initial_config_number=10)
    time.sleep(20)

    
    gptuner_fine = FineStage(
        dbms=dbms, 
        target_knobs_path=target_knobs_path, 
        test=args.test, 
        timeout=args.timeout, 
        seed=args.seed,
        folder=folder
    )

    gptuner_fine.optimize(
        name = f"../{folder}/{args.db}/fine/",
        trials_number=110 # history trials + new tirals
    )   

