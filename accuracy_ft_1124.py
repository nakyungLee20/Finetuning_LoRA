import json

# Open result file and gold file
with open("ft_result_1124_nogold.json", "r") as file:
  results = json.load(file)

with open("train_dataset/test_data_nogold.json", "r") as f:
  gold = json.load(f)


# Preprocess answer
answer=[]
for result in results:
  gen = result.split("[/INST]")[2]
  anws= gen.split("Equation: ")[1]
  ans = anws.split(" </s>")[0]
  ans = ans.rstrip()
  ans = ans.lstrip()
  # print(ans)
  answer.append(ans)

# Compare generated answer with gold
gold_ans = []
for idx in gold:
  gd = idx['text'].split("[/INST]")[1]
  gd = gd.split("Equation: ")[1]
  gd = gd.split(" </s>")[0]
  gd = gd.rstrip()
  gd = gd.lstrip()
  # print(gd)
  gold_ans.append(gd)


corr = 0
wrong = 0
wrong_idx =[]

for i, (res,ans) in enumerate(zip(answer, gold_ans)):
  print(res + "||" + ans)
  print("=======================================================")
  if(res == ans):
    corr += 1
  else:
    wrong += 1
    wrong_idx.append(i)


print(corr)
print(wrong)
# print(corr + wrong)
print(corr/(corr+wrong)*100)

