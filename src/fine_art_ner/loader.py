import json

if __name__ == "__main__":
    with open("datasets/002.txt", "r") as f:
        data = f.read()
        # data = json.loads(data)
        print(data.replace("\n```", "").replace("json", "").replace("```\n", ""))