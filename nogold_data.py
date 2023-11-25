import json

## Load Original Dataset ##
with open("train_dataset/train.json") as file1:
    train = json.load(file1)

with open("train_dataset/test.json") as file2:
    test = json.load(file2)

with open("train_dataset/dev.json") as file3:
    dev = json.load(file3)

## Filter Needed Data ##
# train #
train_data = []
for idx in train:
  q=idx['qa']['question']
  pro=idx['qa']['program']

  pres=idx['pre_text']
  pre_str =""
  for pre in pres:
      pre_str += pre + " "

  posts=idx['post_text']
  post_str =""
  for post in posts:
      post_str += post + " "

  tables = idx['table']
  total=""
  for arg in tables[1:]:
    table =""
    index = len(arg)
    for i in range(1,index):
        arg_str = arg[0] + " " + tables[0][i] + " is " + arg[i] + ". "
        table += arg_str
    total += table + " | "

  fact = pre_str + total + post_str
  text={"Question": fact + "\n" + q, "Answer": pro}
  # print(text)
  train_data.append(text)

# test #
test_data = []
for idx in test:
  q=idx['qa']['question']
  pro=idx['qa']['program']

  pres=idx['pre_text']
  pre_str =""
  for pre in pres:
      pre_str += pre + " "

  posts=idx['post_text']
  post_str =""
  for post in posts:
      post_str += post + " "

  tables = idx['table']
  total=""
  for arg in tables[1:]:
    table =""
    index = len(arg)
    for i in range(1,index):
        arg_str = arg[0] + " " + tables[0][i] + " is " + arg[i] + ". "
        table += arg_str
    total += table + " | "

  fact = pre_str + total + post_str
  text={"Question": fact + "\n" + q, "Answer": pro}
  # print(text)
  test_data.append(text)

# dev #
dev_data = []
for idx in dev:
  q=idx['qa']['question']
  pro=idx['qa']['program']

  pres=idx['pre_text']
  pre_str =""
  for pre in pres:
      pre_str += pre + " "

  posts=idx['post_text']
  post_str =""
  for post in posts:
      post_str += post + " "

  tables = idx['table']
  total=""
  for arg in tables[1:]:
    table =""
    index = len(arg)
    for i in range(1,index):
        arg_str = arg[0] + " " + tables[0][i] + " is " + arg[i] + ". "
        table += arg_str
    total += table + " | "

  fact = pre_str + total + post_str
  text={"Question": fact + "\n" + q, "Answer": pro}
  # print(text)
  dev_data.append(text)


## Process to Chat version ##
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

system_prompt = """Create equations to solve the financial-math problem, using given information. Use ONLY [divide, subtract, add, multiply, greater, exp, min, max] OPERATORS. There are only 2 arguments <arg1>, <arg2> in each operator and are filled with the provided arguments. Use #(step number) as an argument when you want to use previous step's result. Answer in [<Operator>(<arg1>, <arg2>)] FORMAT."""

# Train process #
train_dialogs=[]
for info in train_data:
    dialog = [{"role": "system", "content": system_prompt},
              {"role": "user", "content": info["Question"]},
              {"role": "assistant", "content": "Equation: "+info["Answer"]}]

    train_dialogs.append(dialog)

train_dialog_fix = []
for dialog in train_dialogs:
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )

    for i, (prompt, answer) in enumerate(zip(dialog[::2], dialog[1::2])):
        text = f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
        text = {"text": text}
        train_dialog_fix.append(text)

#for dia in tr_dialog_fix:
#    print(dia)

del train_dialog_fix[3587]
del train_dialog_fix[4790]

with open("train_dataset/train_data_nogold.json", 'w') as outfile:
    json.dump(train_dialog_fix, outfile)
print(len(train_dialog_fix))

# Test Process #
test_dialogs=[]
for info in test_data:
    dialog = [{"role": "system", "content": system_prompt},
              {"role": "user", "content": info["Question"]},
              {"role": "assistant", "content": "Equation: "+info["Answer"]}]

    test_dialogs.append(dialog)

test_dialog_fix = []
for dialog in test_dialogs:
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )

    for i, (prompt, answer) in enumerate(zip(dialog[::2], dialog[1::2])):
        text = f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
        text = {"text": text}
        test_dialog_fix.append(text)

#for dia in test_dialog_fix:
#    print(dia)

with open("train_dataset/test_data_nogold.json", 'w') as outfile:
    json.dump(test_dialog_fix, outfile)
print(len(test_dialog_fix))

# Dev process #
dev_dialogs=[]
for info in dev_data:
    dialog = [{"role": "system", "content": system_prompt},
              {"role": "user", "content": info["Question"]},
              {"role": "assistant", "content": "Equation: "+info["Answer"]}]

    dev_dialogs.append(dialog)

dev_dialog_fix = []
for dialog in dev_dialogs:
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )

    for i, (prompt, answer) in enumerate(zip(dialog[::2], dialog[1::2])):
        text = f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
        text = {"text": text}
        dev_dialog_fix.append(text)

#for dia in dev_dialog_fix:
#    print(dia)

with open("train_dataset/dev_data_nogold.json", 'w') as outfile:
    json.dump(dev_dialog_fix, outfile)

print(len(dev_dialog_fix))
