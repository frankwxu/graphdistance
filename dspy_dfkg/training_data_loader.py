import dspy
import os
import json
import openai


class STIXDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data = self.load_data()

    def load_data(self):
        data = []
        files = os.listdir(self.folder_path)
        text_files = [f for f in files if f.endswith(".txt")]

        for text_file in text_files:
            base_name = os.path.splitext(text_file)[0]
            json_file = f"{base_name}.json"
            if json_file in files:
                with open(
                    os.path.join(self.folder_path, text_file), "r", encoding="utf-8"
                ) as tf:
                    question = tf.read()
                with open(
                    os.path.join(self.folder_path, json_file), "r", encoding="utf-8"
                ) as jf:
                    try:
                        answer = json.load(jf)
                    except json.JSONDecodeError:
                        continue
                data.append((question, answer, text_file))
        return data

    def build_trainset(self):
        train = [
            dspy.Example(question=question, answer=answer).with_inputs("question")
            for question, answer, _ in self.data
        ]
        return train

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BasicQA(dspy.Signature):
    """Analyze the scenario to construct a graph in STIX JSON format with several objects."""

    question = dspy.InputField()
    answer = dspy.OutputField(
        desc="A list of valid json objects. Output must be like '[object1, object2...]'.Each object must have an unique ID."
    )
    # A list include valid json objects.
    # Start with single object. Each STIX object must has to have an ID.


if __name__ == "__main__":
    train_data = STIXDataset("data")
    print(len(train_data))
    print(train_data)
    print(train_data[0])

    # train = [dspy.Example(question=question, answer=answer).with_inputs('question') for question, answer in train_data]
    train = [
        dspy.Example(question=question, answer=answer).with_inputs("question")
        for question, answer, _ in train_data
    ]

    # Path to your API key file
    key_file_path = "openai_api_key.txt"

    # Load the API key from the file
    with open(key_file_path, "r") as file:
        openai_api_key = file.read().strip()

    # Set the API key as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key

    openai.api_key = os.environ["OPENAI_API_KEY"]

    turbo = dspy.OpenAI(model="gpt-3.5-turbo")
    dspy.settings.configure(lm=turbo)

    # Assume dspy.Predict is correctly defined to use a model or some logic to generate answers
    generate_answer = dspy.Predict(BasicQA)

    pred = generate_answer(question=train[2].question)
