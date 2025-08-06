import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--description_path",
    type=str,
    required=True,
    default="/home/haseeb/Desktop/EKAI/ExploringDataLakes/backend/test/compartmentalization/clustering/northwind_descriptions.json"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.description_path) as f: 
        data = json.load(f)
        table_names = list(data["tables"].keys())
        texts = [f"{table_name} : {data["tables"][table_name]['note']}" for table_name in table_names]  

    

if __name__ == "__main__":
    main()