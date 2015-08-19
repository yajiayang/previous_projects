import csv
with open('/Users/royyang/Desktop/synsear.csv', 'rb') as csvfile:
        data = [row for row in csv.reader(csvfile.read().splitlines())]

sum_imp_anto=0
sum_cli_anto=0
sum_imp_syn=0
sum_cli_syn=0
sum_imp_other=0
sum_cli_other=0
count=0
for i,j in enumerate(data[1:]):
    #if count<=1000:'anto' or 
        count+=1
        #print i
        if 'anty' in j[0] or 'anto' in j[0] or 'anto' in j[0] or 'oppo' in j[0] or 'anti' in j[0]:
            sum_imp_anto+=int(j[1])
            sum_cli_anto+=int(j[3])
        elif 'symn' in j[0] or 'sysn' in j[0] or 'syn' in j[0] or'sny' in j[0] or'sino' in j[0] or 'another word' in j[0] or 'other word' in j[0] or 'simi' in j[0]:
            sum_imp_syn+=int(j[1])
            sum_cli_syn+=int(j[3])
        else:
            print j[0]
            sum_imp_other+=int(j[1])
            sum_cli_other+=int(j[3])

print sum_imp_syn
print sum_cli_syn
print float(sum_cli_syn)/sum_imp_syn

print sum_imp_anto
print sum_cli_anto
print float(sum_cli_anto)/sum_imp_anto


print sum_imp_other
print sum_cli_other
print float(sum_cli_other)/sum_imp_other


TotalAmount=[sum_imp_syn,sum_cli_syn,float(sum_cli_syn)/sum_imp_syn]
with open("/Users/royyang/Desktop/Output.txt", "w") as text_file:
    text_file.write("Anto_summary: {0}".format(TotalAmount))


  



