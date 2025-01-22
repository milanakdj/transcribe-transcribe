
text = ""
text_with_timestamp = ""

TEXT_FILE = "complete_timestamp_meeting000.txt"
with open(TEXT_FILE, "r") as f:
    text_with_timestamp = f.read()


list_of_text = text_with_timestamp.split("\n")


print(list_of_text[:4])

for l in list_of_text:
    if('WEBVTT' not in l and l is not None and not(l.startswith('00:'))):
        text = text + "\n"+l

print(text)


with open(f"./no_timestamp_{TEXT_FILE.split('.')[0]}.txt", "w") as f:
    f.write(text)