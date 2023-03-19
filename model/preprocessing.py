data = open("default_2030.csv", "r")
new_data = open("processed_2030.csv", "w")
for line in data:
    new_line = line
    new_line = new_line.lstrip(",")
    new_data.write(new_line)
data.close()
new_data.close()