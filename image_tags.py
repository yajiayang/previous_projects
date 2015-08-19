import re
path = '/Users/royyang/Desktop/imageurl/deep_learning_results.txt'
path1 = '/Users/royyang/Desktop/imageurl/image_tags'
count = 0
with open(path1, 'w') as old:
    with open(path) as new:
        for line in new:
            item = line.replace('\n','').split('\t')
            tag1 = re.sub('[\[\]\(\)\'\"]','',item[3]).split(',')
            tag2 = re.sub('[\[\]\(\)\'\"]','',item[2]).split(',')
            a = float(tag1[1])
            if a >= 0.9:
                count += 1
                print item[1],list(set(list([tag1[0],tag2[0]])))
                old.write(str(item[1])+'\t'+str(tag1[0])+'\t')
                if tag1[0]!=tag2[0]:
                    old.write(str(tag2[0]))
                old.write('\n')
          
tem_path = '/Users/royyang/Desktop/correlate-sauce.csv'         
with open(tem_path) as new:
    for line in new:
        print new


        
        

