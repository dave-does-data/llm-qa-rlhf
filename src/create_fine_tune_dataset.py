from datasets import load_dataset,concatenate_datasets
dataset = load_dataset("databricks/databricks-dolly-15k")

def process_data(dataset):
  append_a = dataset.filter(lambda row: row["category"]=='open_qa')
  append_b = dataset.filter(lambda row: row["category"]=='closed_qa')
  append_c = dataset.filter(lambda row: row["category"]=='general_qa')
  qa_dataset = concatenate_datasets([append_a, append_b, append_c]).shuffle(seed=42)
  qa_dataset = qa_dataset.add_column("text",['### Question: ' +str(row['instruction']) + '### Answer: ' + str(row['response']) for row in qa_dataset])
  return qa_dataset

train = process_data(dataset['train'])
train.to_csv('databricks-dolly-qa-subset-7k.csv')
