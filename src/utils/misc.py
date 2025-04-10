from ruamel.yaml import YAML

def over_write_args_from_file(args, yml_file):
    """
    overwrite arguments according to config file
    """
    if yml_file == '':
        return
    
    yaml = YAML(typ='rt')  # 'rt' is for round-trip parsing (preserves formatting)
    with open(yml_file, "r") as f:
        dic = yaml.load(f)
        for k in dic:
            setattr(args, k, dic[k])
    
        