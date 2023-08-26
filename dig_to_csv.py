import re

queries = []

def get_domain(content_arr):
    line = content_arr[0]
    domain = line.split(" ")[-1]
    return domain

def get_ttl_avg(content_arr):
    isCounting = False
    count = 0
    s = 0
    for e in content_arr:
        if ("ANSWER SECTION" in e):
            isCounting = True
            continue
        if (isCounting and len(e) == 0):
            break
        if (isCounting):
            count += 1
            spl = re.split(" |\t", e)
            n = 0
            for i in range(len(spl)):
                try:
                    n = int(spl[i])
                except ValueError:
                    continue
            # print(n)
            s += n

    return s/count if count != 0 else 0

def get_answer(content_arr):
    isCounting = False
    count = 0
    for e in content_arr:
        if ("ANSWER SECTION" in e):
            isCounting = True
            continue
        if (isCounting and len(e) == 0):
            break
        if (isCounting):
            count += 1
    return count

def get_authority(content_arr):
    isCounting = False
    count = 0
    for e in content_arr:
        if ("AUTHORITY SECTION" in e):
            isCounting = True
            continue
        if (isCounting and len(e) == 0):
            break
        if (isCounting):
            count += 1
    return count

def get_additional(content_arr):
    isCounting = False
    count = 0
    for e in content_arr:
        if ("ADDITIONAL SECTION" in e):
            isCounting = True
            continue
        if (isCounting and len(e) == 0):
            break
        if (isCounting):
            count += 1
    return count


with open("./datasets/fast-flux_dataset/FluXOR_168.95.1_Attack.txt", "r") as file:

    file_lines = file.readlines()

    content = ""

    for l in file_lines:
        line = l

        if ("<<>> DiG" in line):

            content_array = content.split("\n")
            query = {}

            query["content"] = content
            query["domain"] = get_domain(content_array)
            query["answer_count"] = get_answer(content_array)
            query["authority_count"] = get_authority(content_array)
            query["additional_count"] = get_additional(content_array)
            query["ttl_avg"] = get_ttl_avg(content_array)

            queries.append(query)
            content = ""
        
        content += line

for key in queries[1]:
    print(key)

print(queries[1])

with open("out2.csv", "w") as file:
    
    for key in queries[1]:
        if (key == "content"):
            continue
        file.write(key)
        file.write(",")
    file.write("\n")

    for i in range(1, len(queries)):
        for key in queries[i]:
            if (key == "content"):
                continue
            file.write(str(queries[i][key]))
            if (key != "ttl_avg"):
                file.write(",")
        file.write("\n")