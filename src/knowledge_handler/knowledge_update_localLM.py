import json
import os
import textwrap
from typing import Literal
from knowledge_handler.utils import get_hardware_info, get_disk_type
from utils.logger import MyLogger
from knowledge_handler.gpt import LLM
from knowledge_handler.knowledge_transformation_localLM import KGTrans

class KGUpdate(LLM):
    def __init__(self, access_token, knowledge_path, db="postgres", model="llama3-8b"):
        super().__init__(access_token=access_token, model=model)
        self.db = db
        self.knob_num = 0
        self.log= MyLogger("knob_update", knowledge_path, 'INFO').logger
        self.knob_path = knowledge_path # f"./knowledge_collection/{self.db}"
        self._define_path()

    def _define_path(self):
        # Method to set up necessary paths for files
        self.skill_json_path = os.path.join(self.knob_path, "structured_knowledge/normal") # f"./knowledge_collection/{self.db}/structured_knowledge/normal/"
        self.summary_path = os.path.join(self.knob_path, "tuning_lake") # f"./knowledge_collection/{self.db}/tuning_lake"
        self.update_path =  os.path.join(self.knob_path, "knob_info/knob_update.json") # f"./knowledge_collection/{self.db}/knob_info/knob_update.json"

    def read_json_values(self):
        if not os.path.exists(self.update_path) or os.path.getsize(self.update_path) == 0:
            with open(self.update_path, 'w') as file:
                json.dump({}, file)
            return []

        with open(self.update_path, 'r') as file:
            data = json.load(file)
            knob_names = [knob for sublist in data.values() for knob in sublist]
            return knob_names
        
    def update_knob_categories(self, knob, related_knowledge):
        with open(self.update_path, 'r') as file:
            existing_data = json.load(file)

        for k, v in related_knowledge.items():
            if v is True:
                if k in existing_data:
                    existing_data[k].append(knob)
                else:
                    existing_data[k] = [knob]
        
        return existing_data
    
    def merge_data(self, existing_data):
        with open(self.update_path, 'r') as file:
            new_data = json.load(file)

        for key in existing_data:
            if key in new_data:
                combined_list = list(set(existing_data[key] + new_data[key]))
                new_data[key] = combined_list
            else:
                new_data[key] = existing_data[key]
        
        return new_data

    def pipeline(self, knob):
        if knob not in self.read_json_values():
            result = self.filter_knob(knob)
            n = 3
            while "result" not in result.keys() and n > 0:
                self.log.warning(f"Invalid response from filter_knob, re-prompt {n}")
                result = self.filter_knob(knob)
                n -= 1
            if result["result"] is False:
                self.log.info(f"accumulated token:{self.token}, accumulated money:{self.money}")
                return False
        
            # acquire new data
            related_knowledge = self.filter_knowledge(knob)
            print(type(related_knowledge["cpu_related"]))
            # update data
            existing_data = self.update_knob_categories(knob,related_knowledge)
            new_data = self.merge_data(existing_data)
            
            # write the updated data      
            with open(self.update_path, 'w') as file:
                json.dump(new_data, file, indent=4)

        if os.path.exists(os.path.join(self.skill_json_path, knob+".json")) and os.path.getsize(os.path.join(self.skill_json_path, knob+".json")) != 0:
            self.log.info(f"Already finished to update structured knowledge for {knob}, skip.")
            self.log.info(f"accumulated token:{self.token}, accumulated money:{self.money}")
            return False

        new_structure = self.update_knowledge(knob)
        if new_structure is False:
            self.log.info(f"accumulated token:{self.token}, accumulated money:{self.money}")
            return False 
        with open(os.path.join(self.skill_json_path, knob+".json"), 'w') as file:
            json.dump(new_structure, file)
        print(f"Finished to update structured knowledge for {knob}")
        self.log.info(f"accumulated token:{self.token}, accumulated money:{self.money}")
        return new_structure

    # offline
    def filter_knob(self, knob):

        # knob -> cpu, ram, disk_size, disk_type 
        # LLM: True of False

        prompt = textwrap.dedent(f"""
        I first give you a knob of {self.db}, determine if it is related to resources, focusing primarily on CPU, RAM, disk size, and disk type. Note that some knobs may not appear directly related to resources but are indeed associated with them, so please exercise careful discernment. 
            
        let's think step by step

        step 1: Summarize the function of  knob from {self.db}  with no more than five sentences.
        step 2: Judge whether this knob is related to cpu, ram, disk type or disk size.
        step 3: If the knob is related to any hardware resource in step 2, return the boolean value true, otherwise, return the boolean value false.

        Please give me the result in json format.
      
        KNOB:
        {knob}         

        JSON RESULT TEMPLATE:
        {{
            "result" : // Set as Boolean true if resource-related, otherwise false
        }}
        """    
        )
        self.log.info(f"filter_knob - prompt - {knob}: {prompt}")
        response = self.get_GPT_response_json(prompt)
        self.log.info(f"filter_knob - response - {knob}: {response}")
        self.token += self.calc_token()
        print(json.dumps(response, indent=4))
        return response

    # offline
    def filter_knowledge(self, knob):

        file_name = f'{knob}.txt'
        file_path = os.path.join(self.summary_path, file_name)

        try:
            with open(file_path, 'r') as summary_file:
                tuning_lake_doc = summary_file.read()

        except FileNotFoundError:
            print(f"File {file_name} not found in path {self.summary_path}.")
        except Exception as e:
            print(f"An error occurred while opening the file: {e}")

        prompt = textwrap.dedent(f"""
            I first give you a knob of {self.db} and its tuning suggestion, please judge whether the tuning suggestion is related to the given hardware sources.Note that a knob may be related to more than one class.

            KNOB:
            {knob}
            TUNING_SUGGESTION:
            {tuning_lake_doc}
            
            Now think step by step, and give me the result in json format. If the suggestion is related to the resource, put true as the value. If not, return false.
            JSON RESULT TEMPLATE:
            {{
                "cpu_related": // Set as Boolean true if CPU-related, otherwise false
                "ram_related": // Set as Boolean true if RAM-related, otherwise false
                "disk_size_related": // Set as Boolean true if disk size-related, otherwise false
                "disk_type_related": // Set as Boolean true if disk type-related, otherwise false
            }}

            """    
        )
        self.log.info(f"filter_knowledge - prompt - {knob}: {prompt}")
        response = self.get_GPT_response_json(prompt)
        self.log.info(f"filter_knowledge - response - {knob}: {response}")
        self.token += self.calc_token()
        print(json.dumps(response, indent=4))
        return response

    def read_knobs_from_json(self, type: Literal['cpu_related', 'ram_related', 'disk_size_related', 'disk_type_related']):

        with open(self.update_path, 'r') as file:
            data = json.load(file)

        knobs = data.get(type, []) 
        return knobs
    
    # online
    def update_knowledge(self, knob):

        try:
            with open(os.path.join(self.skill_json_path, knob+".json"), 'r') as file:
                structured_knowledge = json.load(file)
        except:
            print(f"The structured knowledge of {knob} is empty, generate the structured knowledge first.")
            raise         
     
        new_cpu, new_ram, new_disk_size = get_hardware_info()
        new_disk_type = get_disk_type()
        new_hardware = {
            "cpu": new_cpu,
            "ram": new_ram,
            "disk_size": new_disk_size,
            "disk_type": new_disk_type
        }

        is_knob_matched = False
        for i in ["cpu_related", "ram_related", "disk_size_related", "disk_type_related"]:
            hardware_key = "".join(i.split("_")[:-1])
            if knob in self.read_knobs_from_json(i) and structured_knowledge.get(hardware_key) != new_hardware.get(hardware_key):
                is_knob_matched = True
                knowledge_trans = KGTrans(db=self.db, api_base=self.api_base, api_key=self.api_key)
                return knowledge_trans.vote(knob)

        if not is_knob_matched:
            return False